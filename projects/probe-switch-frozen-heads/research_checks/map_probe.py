"""Readability map (Iris, T2; PREREG 'Predicted numbers from the map' + v1.1).

Can ONE position's residual tell 'probe attached' from 'not attached'? Single rows, not pairs (the switch decides from
one forward pass). Fit on the 512 TRAIN prompts x 2 conditions, score on the 128 TEST prompts x 2. fp32 CPU from the
bf16 states. Probes (frozen in PREREG): logistic regression (standardised, C=1) and a 2-layer MLP probe (64 hidden,
early stopping, width >= 16). Null: 200 random within-pair label flips of the TEST rows, AUROC recomputed on the same
scores (exact permutation null for held-out AUROC under 'condition not readable'); p95 reported per cell.

Cells:
  eager column (map_states.py): pos 79 at in20..in27, mid20..mid27, final  -> logistic + MLP
                                pos 70..78 at in20..in26                   -> logistic (route rule source)
  whole-graph column (cache.py pass B): pos 71 and 79 at in14..in27, mid14..mid27, final -> logistic; pos 79 also MLP
Predicted MLP-27 strict switch = max(0, 2*acc - 1), acc = larger held-out row accuracy of the two probes at eager mid27
pos 79 (Kasper identity: switch - reverse switch = 2*acc - 1 for any single-row detector).

    PY map_probe.py <project dir> <map dir with eager_states.npy> <out.json> [--quick]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(1027)


def f32(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def null_p95(s, n, k=200):
    """s = scores of [n unprobed test rows, n probed test rows] (row j and j+n = same prompt)."""
    y = np.r_[np.zeros(n), np.ones(n)]
    out = []
    for _ in range(k):
        flip = RNG.random(n) < 0.5
        yy = y.copy(); yy[:n][flip] = 1; yy[n:][flip] = 0
        out.append(roc_auc_score(yy, s))
    return float(np.percentile(out, 95))


def fit(X0tr, X1tr, X0te, X1te, kind):
    Xtr = np.concatenate([X0tr, X1tr]); Xte = np.concatenate([X0te, X1te])
    ytr = np.r_[np.zeros(len(X0tr)), np.ones(len(X1tr))]
    yte = np.r_[np.zeros(len(X0te)), np.ones(len(X1te))]
    if kind == "logistic":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    else:
        m = make_pipeline(StandardScaler(), MLPClassifier((64,), early_stopping=True, max_iter=500, random_state=0))
    m.fit(Xtr, ytr)
    s = m.predict_proba(Xte)[:, 1]
    n = len(X0te)
    r = {"auroc": float(roc_auc_score(yte, s)), "acc": float(np.mean((s > 0.5) == yte)),
         "paired_win": float(np.mean(s[n:] > s[:n])), "train_acc": float(m.score(Xtr, ytr)),
         "null_p95": null_p95(s, n)}
    r["readable"] = bool(r["auroc"] > r["null_p95"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proj"); ap.add_argument("mapdir"); ap.add_argument("out")
    ap.add_argument("--quick", action="store_true", help="pos 79 eager only (smoke)")
    a = ap.parse_args()
    proj, mdir = Path(a.proj), Path(a.mapdir)
    t0 = time.time()
    em = json.load(open(mdir / "eager_meta.json"))
    E = np.load(mdir / "eager_states.npy", mmap_mode="r")  # [N, 2, 10, 17, H]
    pid = em["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(proj / "raw" / "prompts.jsonl")}
    tr = [i for i, p in enumerate(pid) if split[p] == "train"]
    te = [i for i, p in enumerate(pid) if split[p] == "test"]
    res = {"n_train": len(tr), "n_test": len(te), "eager": {}, "whole": {}, "eager_sites": em["sites"]}
    print("train", len(tr), "test", len(te), flush=True)

    def cell_e(pos, site, kinds):
        p, s = em["positions"].index(pos), em["sites"].index(site)
        X = lambda ids, c: f32(E[ids, c, p, s])
        d = {}
        for k in kinds:
            d[k] = fit(X(tr, 0), X(tr, 1), X(te, 0), X(te, 1), k)
        a0, a1 = X(te, 0), X(te, 1)
        d["rel_diff"] = float((np.linalg.norm(a1 - a0, axis=-1) / np.linalg.norm(a0, axis=-1)).mean())
        d["share_prompts_any_bit_differs"] = float(np.mean((a1 != a0).any(-1)))
        res["eager"][f"{site}@{pos}"] = d
        print("eager", site, pos, {k: (round(d[k]["auroc"], 3), round(d[k]["acc"], 3), round(d[k]["null_p95"], 3))
                                   for k in kinds}, f"{time.time() - t0:.0f}s", flush=True)

    for site in em["sites"]:
        cell_e(79, site, ("logistic", "mlp"))
    if not a.quick:
        for l in range(20, 27):
            for pos in range(70, 79):
                cell_e(pos, f"in{l}", ("logistic",))
        bdir = proj / "raw" / "cache"
        mb = json.load(open(bdir / "meta_B.json"))
        B = [np.load(bdir / f"{c}_B_states.npy", mmap_mode="r") for c in ("none", "probe")]  # [N, 2pos, sites, H]
        assert mb["prompt_ids"] == pid
        for si, site in enumerate(mb["sites"]):
            for pi, pos in enumerate(mb["positions"]):
                kinds = ("logistic", "mlp") if pos == 79 else ("logistic",)
                X = lambda ids, c: f32(B[c][ids, pi, si])
                d = {k: fit(X(tr, 0), X(tr, 1), X(te, 0), X(te, 1), k) for k in kinds}
                a0, a1 = X(te, 0), X(te, 1)
                d["rel_diff"] = float((np.linalg.norm(a1 - a0, axis=-1) / np.linalg.norm(a0, axis=-1)).mean())
                d["share_prompts_any_bit_differs"] = float(np.mean((a1 != a0).any(-1)))
                res["whole"][f"{site}@{pos}"] = d
                print("whole", site, pos, {k: round(d[k]["auroc"], 3) for k in kinds}, f"{time.time() - t0:.0f}s",
                      flush=True)
    m27 = res["eager"]["mid27@79"]
    acc = max(m27["logistic"]["acc"], m27["mlp"]["acc"])
    res["prediction_MLP27"] = {"acc": acc, "predicted_strict_switch": max(0.0, 2 * acc - 1),
                               "tolerance": 0.10, "bit_ceiling": m27["share_prompts_any_bit_differs"]}
    res["clause1_max_logistic_auroc_in20_27@79"] = max(res["eager"][f"in{l}@79"]["logistic"]["auroc"]
                                                       for l in range(20, 28))
    if not a.quick:
        src = {k: v for k, v in res["eager"].items() if k.split("@")[0] in {f"in{l}" for l in range(20, 27)}
               and int(k.split("@")[1]) in range(70, 79)}
        best = max(src.items(), key=lambda kv: kv[1]["logistic"]["auroc"])
        res["route_source"] = {"best_cell": best[0], "auroc": best[1]["logistic"]["auroc"],
                               "null_p95": best[1]["logistic"]["null_p95"],
                               "readable_cells": sorted(k for k, v in src.items() if v["logistic"]["auroc"] > 0.80
                                                        and v["logistic"]["auroc"] > v["logistic"]["null_p95"])}
        res["clause1_max_logistic_auroc_in20_27@79_whole"] = max(res["whole"][f"in{l}@79"]["logistic"]["auroc"]
                                                                 for l in range(20, 28))
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print("PRED MLP-27", res["prediction_MLP27"], "clause1", res["clause1_max_logistic_auroc_in20_27@79"],
          res.get("route_source"), flush=True)


if __name__ == "__main__":
    main()
