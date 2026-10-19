"""Draw-luck check: does the PREREG s8 point gap (base minus policy refit AUROC at n=200, mean over 20 draws)
depend on which 20 class-balanced label draws were taken? 20 draw seeds x 20 draws each, own acts, judge labels.
Also the same on the judge-and-string-agree TEST subset (robustness guard).

    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_seedsweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, balanced_draw, fit_lr, score  # noqa: E402
from vera_rebuild import LAM1, load  # noqa: E402

OUT = Path(__file__).resolve().parent
N, NDRAW, NSEED = 200, 20, 20


def per_seed(d, seed):
    rng = np.random.default_rng(seed)
    te, po = d["split"] == "test", d["split"] == "pool"
    ypool, yt = d["y"][po], d["y"][te]
    clean = (yt >= 0) & (yt == d["ys"][te])
    full, sub = [], []
    for _ in range(NDRAW):
        idx = balanced_draw(ypool, N, rng)
        s = score(fit_lr(d["own"][po][idx], ypool[idx]), d["own"][te])
        full.append(auroc(yt, s))
        sub.append(auroc(yt[clean], s[clean]))
    return np.mean(full), np.mean(sub)


def main():
    data = {p: load(p) for p in ["base"] + LAM1}
    vals = {p: np.array([per_seed(d, 100 + k) for k in range(NSEED)]) for p, d in data.items() if d is not None}
    res = {}
    for p in [q for q in LAM1 if q in vals]:
        g = vals["base"] - vals[p]  # paired by seed index only for bookkeeping; seeds are independent per policy
        res[p] = {
            "gap_full_mean": float(g[:, 0].mean()), "gap_full_min": float(g[:, 0].min()), "gap_full_max": float(g[:, 0].max()),
            "share_seeds_gap_full_gt_0.05": float((g[:, 0] > 0.05).mean()),
            "gap_clean_mean": float(g[:, 1].mean()), "gap_clean_min": float(g[:, 1].min()), "gap_clean_max": float(g[:, 1].max()),
            "share_seeds_gap_clean_gt_0.05": float((g[:, 1] > 0.05).mean()),
        }
        print(p, json.dumps({k: round(v, 4) for k, v in res[p].items()}))
    res["refit200_per_seed"] = {p: v.tolist() for p, v in vals.items()}
    (OUT / "vera_seedsweep.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
