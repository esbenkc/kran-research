"""Iris: saved check of the KL-cap formula in THEORY.md item 2 (Kasper #19376 asked for runs on disk).
Same toy as toy_real.py; varies SIGMA (prompt spread, isotropic, so also along the cue) and BETA (KL weight) at two
sub-edge alphas. Late gap = mean dev gap over the last half of the budget (curve points every 512 updates), minus gap0.
Run: uvx --with numpy python toy_sens.py  -> toy_sens.json"""
import json
import math
from pathlib import Path

import numpy as np

import toy_real as t

BUDGET = 20480
rows = []
for alpha in (0.0554, 0.0979):
    for sigma, beta in ((0.35, 1.0), (0.7, 1.0), (1.4, 1.0), (0.7, 0.5), (0.7, 2.0)):
        for seed in (0, 1):
            t.SIGMA, t.BETA = sigma, beta
            W = t.world(0)
            o = t.run(W, alpha, seed, budget=BUDGET)
            c = [g for (s, g, *_ ) in o["curve"] if s >= BUDGET // 2]
            rows.append(dict(alpha=alpha, sigma=sigma, beta=beta, seed=seed, T=o["T"],
                             late_gap=(float(np.mean(c)) - o["gap0"]) if c else None))
            print(rows[-1], flush=True)
t.SIGMA, t.BETA = 0.7, 1.0


def slope(key, fixed):
    out = {}
    for alpha in (0.0554, 0.0979):
        pts = [(r[key], r["late_gap"]) for r in rows if r["alpha"] == alpha and all(r[k] == v for k, v in fixed.items())
               and r["late_gap"] is not None and r["T"] is None]
        if len({p[0] for p in pts}) >= 2:
            x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
            out[str(alpha)] = round(float(np.polyfit(x, y, 1)[0]), 2)
    return out


summ = dict(late_gap_slope_vs_sigma_at_beta1=slope("sigma", {"beta": 1.0}),
            late_gap_slope_vs_beta_at_sigma07=slope("beta", {"sigma": 0.7}),
            note="slopes use only censored runs (a run that crossed 0.10 has no plateau); T given where it crossed")
json.dump(dict(budget=BUDGET, rows=rows, summary=summ), open(Path(__file__).with_name("toy_sens.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
