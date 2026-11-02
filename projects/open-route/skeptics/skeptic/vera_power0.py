"""Vera pre-data power sim for the prereg ratio R = key(open)/key(blocked), pooled stronger five.
Anchors: per-model blocked key rates from warning-override off_absent (my own tags, 40 runs each).
Bootstrap: resample runs within model x arm, B=2000 per sim (percentile 95% CI)."""
import numpy as np, sys
blocked = np.array([1.0, 0.875, 1.0, 0.975, 0.75])  # haiku, g3.1, g3.5, mini, nano
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rng = np.random.default_rng(1)
S, B = 400, 2000
print(f"n per model per cell = {n}; blocked anchor mean {blocked.mean():.3f}")
print("true_ratio  P(Rlo>0.35&R>=0.5)  P(Rhi<0.5)  mean CI width")
for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    hold = fail = 0; w = []
    for s in range(S):
        kb = rng.binomial(n, blocked); ko = rng.binomial(n, np.clip(blocked * r, 0, 1))
        R = ko.sum() / kb.sum()
        bb = rng.binomial(n, kb / n, size=(B, 5)).sum(1); bo = rng.binomial(n, ko / n, size=(B, 5)).sum(1)
        Rb = bo / np.maximum(bb, 1); lo, hi = np.percentile(Rb, [2.5, 97.5])
        hold += (lo > 0.35) and (R >= 0.5); fail += hi < 0.5; w.append(hi - lo)
    print(f"{r:9.2f}  {hold/S:18.2f}  {fail/S:10.2f}  {np.mean(w):.3f}")
