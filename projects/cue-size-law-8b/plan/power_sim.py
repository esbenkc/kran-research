"""Felix: design check of the frozen seed-level rule (no data; parent per-seed slopes only).
Rule: per-seed OLS slope, t-interval over n seeds (df n-1); CONFIRMED if inside [-1.4, -0.7], REJECTED if outside.
Per-seed slope SD from the parent: frozen grid -0.853 / -0.948 (SD 0.067); dense T -0.863 / -0.941 (SD 0.055).
Run: python power_sim.py  (prints P(inside) and P(outside) per true slope, seed SD, seed count)."""
import numpy as np
from scipy import stats

LO, HI = -1.4, -0.7
rng = np.random.default_rng(0)
print("n  true   " + "  ".join(f"sd{sd}:in/out" for sd in (0.055, 0.067, 0.10, 0.15)))
for n in (3, 4, 5):
    t = stats.t.ppf(0.975, n - 1)
    for mu in (-0.5, -0.8, -0.9, -1.0, -1.2, -1.3, -1.7, -2.0):
        cells = []
        for sd in (0.055, 0.067, 0.10, 0.15):
            s = rng.normal(mu, sd, (100_000, n))
            m, h = s.mean(1), t * s.std(1, ddof=1) / np.sqrt(n)
            cells.append(f"{(((m - h) >= LO) & ((m + h) <= HI)).mean():.2f}/{(((m + h) < LO) | ((m - h) > HI)).mean():.2f}")
        print(f"{n}  {mu:+.1f}  " + "  ".join(cells))
