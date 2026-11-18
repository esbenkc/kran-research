"""Quadratic-in-ln-share fit over every scored main-arm run of each model (theory-1118 continuation, Iris).

Not a frozen read. Source of the post's foldout sentence: "A quadratic in ln share ... curvature t = ... Its local
factor per halving runs from x... at the largest cue to x... at the smallest".
Fit: ln T = c2 (ln s)^2 + c1 ln s + c0 by OLS. t = c2 / se(c2) with the usual OLS covariance (n - 3 dof).
Local slope at share s: 2 c2 ln s + c1; local factor per halving = 2^(-local slope).
Runs: exactly the 30 cells analysis.json scores per model: 8B mini seeds 0-2 (m8_*), laptop seeds 3-4 (l8_*);
0.6B rerun seeds 0-4 (runs_06b/m06_*). Share = the run's own measured share (r["share"]).
Run: python3 curvature_1118.py  ->  data/curvature_1118.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "qwen3-8b": [f"data/runs/{'m8' if s < 3 else 'l8'}_k{k}_s{s}.json" for s in range(5) for k in range(6)],
    "qwen3-0.6b": [f"data/runs_06b/m06_k{k}_s{s}.json" for s in range(5) for k in range(6)],
}


def fit(paths):
    x, y = [], []
    for p in paths:
        r = json.load(open(os.path.join(HERE, p)))
        assert r["T_dense"] and not r["censored"], p
        x.append(np.log(r["share"]))
        y.append(np.log(r["T_dense"]))
    x, y = np.array(x), np.array(y)
    n = len(x)
    A = np.c_[x**2, x, np.ones(n)]
    c, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    s2 = float(res[0]) / (n - 3)
    cov = s2 * np.linalg.inv(A.T @ A)
    t = c[0] / np.sqrt(cov[0, 0])
    lo, hi = float(np.exp(x.max())), float(np.exp(x.min()))
    loc = {s: 2 * c[0] * np.log(s) + c[1] for s in (lo, hi)}
    return dict(n=n, c2=float(c[0]), c1=float(c[1]), c0=float(c[2]), se_c2=float(np.sqrt(cov[0, 0])), t_c2=float(t),
                largest_share=lo, smallest_share=hi,
                local_slope_largest=float(loc[lo]), local_slope_smallest=float(loc[hi]),
                factor_per_halving_largest=float(2 ** -loc[lo]), factor_per_halving_smallest=float(2 ** -loc[hi]))


out = {m: fit(p) for m, p in SETS.items()}
json.dump(out, open(os.path.join(HERE, "data/curvature_1118.json"), "w"), indent=1)
for m, v in out.items():
    print(m, f"n {v['n']} c2 {v['c2']:.4f} t {v['t_c2']:.2f} x{v['factor_per_halving_largest']:.3f} -> "
             f"x{v['factor_per_halving_smallest']:.3f}")
