"""Vera's rebuild of build's numbers from build's saved per-row scores (data/fits/<read>.json) + the labels.
(1) per-model AUROCs for every arm with my own AUROC; (2) PREREG s6 verdict (m, g, CIs, status) with my own paired
prompt bootstrap, for (c) and (c2) if present, on fact-stated AND all-truthful negatives; (3) exact refit of arm (a)
from build's saved draws on my own layer-12 means (vera_feats) -> max |score diff|; (4) positive control.
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_rebuild.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import MODELS, TRAINED, auroc, classes, load_labels  # noqa: E402
from vera_io import D20, P  # noqa: E402
from vera_verdict import verdict  # noqa: E402

HERE = Path(__file__).resolve().parent
FITS = P / "data" / "fits"
OUT = {}


def load_fit(model, read="own"):
    f = FITS / f"{model}_{read}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    L = load_labels(model, D20)[:1000]
    assert d["prompt_ids"] == [r["prompt_id"] for r in L], f
    out = {}
    for k, v in d["scores"].items():
        a = np.array(v, float)
        out[k] = np.nanmean(a, 1) if a.ndim == 2 else a  # OOF score = mean over the row's draws
    for suf, ren in [("b2", lambda k: k.replace("b_", "b2_", 1)), ("c2", lambda k: k)]:  # D7 refit of b; D1 arm c2
        g = FITS / f"{model}_{read}_{suf}.json"
        if g.exists():
            e = json.load(open(g))
            assert e.get("prompt_ids", d["prompt_ids"]) == d["prompt_ids"], g
            for k, v in e["scores"].items():
                out[ren(k)] = np.nanmean(np.array(v, float), 1)
    return out, L


S, LAB = {}, {}
for m in MODELS:
    r = load_fit(m)
    if r is None:
        print(m, "no fit file yet"); continue
    S[m], LAB[m] = r
    c = classes(LAB[m])
    row = {"n_spin": int(c["spin"].sum()), "n_truth_fs": int(c["truthful_fs"].sum()), "n_truth": int(c["truthful"].sum())}
    for arm, s in S[m].items():
        row[arm] = {"lies": auroc(s[c["lie"]], s[c["truthful"]]), "spin_fs": auroc(s[c["spin"]], s[c["truthful_fs"]]),
                    "spin_all": auroc(s[c["spin"]], s[c["truthful"]])}
    OUT[m] = row
    print(m, row["n_spin"], row["n_truth_fs"], " | ".join(f"{a}: lies {v['lies']:.3f} spin_fs {v['spin_fs']:.3f} spin_all {v['spin_all']:.3f}"
                                                         for a, v in row.items() if isinstance(v, dict)))

# exact refit of arm a from build's draws
draws = json.load(open(P / "data" / "draws.json")); folds = json.load(open(P / "data" / "folds.json"))
for m in S:
    m12 = np.load(HERE / "feats" / f"{m}_own.npz")["meanL"][:1000, 11].astype(np.float64)
    y = classes(LAB[m])["lie"].astype(int); fr = np.array([folds[r["prompt_id"]] for r in LAB[m]])
    sa = np.zeros(1000)
    for f, dl in draws[m].items():
        te = np.flatnonzero(fr == int(f))
        for idx in dl:
            sc = StandardScaler().fit(m12[idx])
            lr = LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced").fit(sc.transform(m12[idx]), y[idx])
            sa[te] += lr.decision_function(sc.transform(m12[te])) / len(dl)
    ba = S[m]["a_all"]; c = classes(LAB[m])
    print(m, "exact refit a: corr %.6f max|diff| %.4g | my AUROC lies %.4f vs build %.4f | spin_fs %.4f vs %.4f" % (
        np.corrcoef(sa, ba)[0, 1], np.abs(sa - ba).max(), auroc(sa[c["lie"]], sa[c["truthful"]]), auroc(ba[c["lie"]], ba[c["truthful"]]),
        auroc(sa[c["spin"]], sa[c["truthful_fs"]]), auroc(ba[c["spin"]], ba[c["truthful_fs"]])))
    OUT[m]["exact_a_maxdiff"] = float(np.abs(sa - ba).max())

if "base" in S:
    c = classes(LAB["base"])
    pa = auroc(S["base"]["a_all"][c["lie"]], S["base"]["a_all"][c["truthful"]])
    for arm in ["c_all", "c2_all"]:
        if arm in S["base"]:
            pc = auroc(S["base"][arm][c["lie"]], S["base"][arm][c["truthful"]])
            print(f"POSITIVE CONTROL base all lies: {arm} {pc:.4f} vs a {pa:.4f} -> {'PASS' if pc > pa else 'FAIL'}")

if all(m in S for m in TRAINED):
    for arm in ["c_all", "c2_all"]:
        if not all(arm in S[m] for m in TRAINED):
            continue
        for neg in ["truthful_fs", "truthful"]:
            v = verdict({m: {"c": S[m][arm], "a": S[m]["a_all"]} for m in TRAINED}, {m: LAB[m] for m in TRAINED}, neg=neg)
            print(f"VERDICT {arm} neg={neg}: m {v['m']:.4f} [{v['m_lo']:.4f},{v['m_hi']:.4f}] g {v['g']:+.4f} [{v['g_lo']:+.4f},{v['g_hi']:+.4f}] "
                  f"status {v['status']} per-model " + " ".join(f"{k} {x:.3f}" for k, x in v["per_model"].items()))
            OUT[f"verdict_{arm}_{neg}"] = {k: v[k] for k in ["m", "m_lo", "m_hi", "g", "g_lo", "g_hi", "status", "per_model"]}
json.dump(OUT, open(HERE / "vera_rebuild.json", "w"), indent=1)
