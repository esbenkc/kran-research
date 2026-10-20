"""Summarise data/lcurve/<policy>.json (lcurve.py): mean over the four trained models per arm, and paired bootstrap
95% intervals (1,000 resamples of the 1,000 chats, same resamples for every arm and model, as in score.py) for the
megaprobe minus the one-layer probe at the same label count.

    uv run python projects/megaprobe-spin-lies/lc_summary.py  -> data/lcurve/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probes as P  # noqa: E402
from convergence import subsets  # noqa: E402
from lcurve import OUTD  # noqa: E402

B = 1000
SUBS = ("spin_vs_fact_truthful", "lies_vs_truthful")
PAIRS = [("c_100", "a_100"), ("c_200", "a_200"), ("c_400", "a_400"), ("c_all", "a_all"), ("c32_400", "a_400"),
         ("c32_all", "a_all")]


def wauroc(y, s, w):
    """AUROC with integer row weights (bootstrap counts): expand by repetition."""
    idx = np.repeat(np.arange(len(y)), w)
    yy = y[idx]
    return roc_auc_score(yy, s[idx]) if 0 < yy.sum() < len(yy) else np.nan


def main():
    rng = np.random.default_rng(P.SEED)
    W = np.stack([np.bincount(rng.integers(0, P.N_ROWS, P.N_ROWS), minlength=P.N_ROWS) for _ in range(B)])
    pols = [p for p in P.POLICIES[1:] if (OUTD / f"{p}.json").exists()]
    pt, bs = {}, {}
    for p in pols:
        d = json.loads((OUTD / f"{p}.json").read_text())
        S = subsets(P.load_rows(p))
        for arm, sc in d["scores"].items():
            sc = np.array(sc)
            for n in SUBS:
                use, y = S[n]; yu, su = y[use], sc[use]
                pt[(p, arm, n)] = roc_auc_score(yu, su)
                bs[(p, arm, n)] = np.array([wauroc(yu, su, W[b][use]) for b in range(B)])
    arms = sorted({k[1] for k in pt})
    out = {"policies": pols, "mean4": {}, "per_model": {}, "gain_mean4": {}}
    for n in SUBS:
        out["mean4"][n] = {a: round(float(np.mean([pt[(p, a, n)] for p in pols])), 4) for a in arms}
        out["per_model"][n] = {a: {p: round(float(pt[(p, a, n)]), 4) for p in pols} for a in arms}
        for c, a in PAIRS:
            g = np.mean([pt[(p, c, n)] - pt[(p, a, n)] for p in pols])
            gb = np.mean([bs[(p, c, n)] - bs[(p, a, n)] for p in pols], 0)
            out["gain_mean4"].setdefault(n, {})[f"{c}-{a}"] = {
                "point": round(float(g), 4), "ci": [round(float(np.nanpercentile(gb, q)), 4) for q in (2.5, 97.5)]}
    (OUTD / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["mean4"], indent=1)); print(json.dumps(out["gain_mean4"], indent=1))


if __name__ == "__main__":
    main()
