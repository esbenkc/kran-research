"""Vera: power of the frozen 8B status rule, before any 8B data.

Rule simulated: per-seed OLS slope of log T on log x (5 or 6 halvings), seed-level 95% t-interval over 3 seeds (df 2).
CONFIRMED-like = interval inside [-1.4, -0.7]; REJECTED-like = interval entirely outside the band.
Noise: iid per-run log-T noise sigma (parent: 0.10 interpolated T, 0.15 frozen-grid T).
Grid: 'frozen' rounds T up to the next frozen eval (8,16,24,32,48,64, then every 64); 'dense' rounds up to 8; 'interp' none.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(1116)
GRID = np.array([8, 16, 24, 32, 48, 64] + list(range(128, 65536, 64)))


def quant(T, grid):
    if grid == "interp":
        return T
    if grid == "dense":
        return np.ceil(T / 8) * 8
    idx = np.searchsorted(GRID, T)
    return GRID[np.minimum(idx, len(GRID) - 1)]


def sim(b, sigma, n_sizes, T_top, grid, n_seeds=3, reps=4000):
    lx = -np.log(2) * np.arange(n_sizes)  # log x, top size at 0
    conf = rej = 0
    lo_hi = []
    for _ in range(reps):
        slopes = []
        for s in range(n_seeds):
            T = T_top * np.exp(b * lx + rng.normal(0, sigma, n_sizes))
            T = quant(np.maximum(T, 1), grid)
            slopes.append(np.polyfit(lx, np.log(T), 1)[0])
        m, sd = np.mean(slopes), np.std(slopes, ddof=1)
        h = stats.t.ppf(0.975, n_seeds - 1) * sd / np.sqrt(n_seeds)
        lo, hi = m - h, m + h
        if lo >= -1.4 and hi <= -0.7:
            conf += 1
        elif hi < -1.4 or lo > -0.7:
            rej += 1
    return conf / reps, rej / reps


print("P(interval inside band) / P(interval outside band)")
for grid in ("interp", "dense", "frozen"):
    for T_top in (30, 200):
        for sigma in (0.10, 0.15, 0.25):
            row = []
            for b in (-0.5, -0.7, -0.9, -1.1, -1.4, -2.0):
                c, r = sim(b, sigma, 5, T_top, grid)
                row.append(f"b{b}:{c:.2f}/{r:.2f}")
            print(f"{grid:6s} Ttop={T_top:3d} sig={sigma:.2f} 5 sizes | " + " ".join(row))
# 6 sizes, sigma .15, interp
for n in (5, 6, 7):
    row = []
    for b in (-0.7, -0.9, -1.4, -2.0):
        c, r = sim(b, 0.15, n, 100, "dense")
        row.append(f"b{b}:{c:.2f}/{r:.2f}")
    print(f"dense Ttop=100 sig=0.15 {n} sizes | " + " ".join(row))
