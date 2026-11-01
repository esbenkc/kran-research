"""After-training answer-token map (PREREG 'Mechanism readouts'): for one trained model, held-out logistic AUROC at
position 79 (answer) and 71 (<|im_end|>) for every block input 20-27, mid-block 20-27 and the final residual.
Same probe as research_checks/map_probe.py (standardised logistic regression, C=1, max_iter 2000; fit on 512 TRAIN
prompts x 2 conditions, scored on 128 TEST prompts x 2), fp32 CPU from the bf16 states that audit.py saved.
Null: 200 within-pair label flips of the TEST rows on the same scores, p95 per cell.

    PY after_map.py --run mlp8_none_s0      (CPU, <1 GB)
writes data/after_map/<run>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent


def f32(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-null", type=int, default=200)
    a = ap.parse_args()
    au = json.load(open(HERE / "data" / "audit" / f"{a.run}.json"))
    st = np.load(HERE / au["states"]["path"], mmap_mode="r")  # [704, 2 cond, 2 pos, sites, H]
    pid = json.load(open(HERE / "raw" / "cache" / "meta_A.json"))["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    tr = [i for i, p in enumerate(pid) if split[p] == "train"]
    te = [i for i, p in enumerate(pid) if split[p] == "test"]
    rng = np.random.default_rng(1027)
    res = {"run": a.run, "n_train": len(tr), "n_test": len(te), "cells": {}}
    for pi, pos in enumerate(au["states"]["positions"]):
        for si, site in enumerate(au["states"]["sites"]):
            X = f32(st[:, :, pi, si])  # [704, 2, H]
            Xtr = np.concatenate([X[tr, 0], X[tr, 1]]); ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
            Xte = np.concatenate([X[te, 0], X[te, 1]]); yte = np.r_[np.zeros(len(te)), np.ones(len(te))]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
            s = m.predict_proba(Xte)[:, 1]
            null = []
            for _ in range(a.n_null):
                flip = rng.integers(0, 2, len(te)).astype(bool)
                y = yte.copy(); y[:len(te)][flip] = 1; y[len(te):][flip] = 0
                null.append(roc_auc_score(y, s))
            res["cells"][f"{site}@{pos}"] = {"auroc": float(roc_auc_score(yte, s)),
                                             "acc": float(np.mean((s > 0.5) == yte)),
                                             "null_p95": float(np.percentile(null, 95))}
        print(pos, {k: round(v["auroc"], 3) for k, v in res["cells"].items() if k.endswith(f"@{pos}")}, flush=True)
    (HERE / "data" / "after_map").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "after_map" / f"{a.run}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
