"""Null calibration for rule B (Kasper #9003): >=1000 synthetic no-effect ladders; one-sided miss rate of the
t-interval (K-2 df, 95%) for I (d_m on x) and for the static change (u_static on x)."""
import numpy as np
from scipy import stats

def ladder(rng, K, n=40, nb=20, sd_o=0.1, sd_g=0.05, touch=0.6, base=0.6):
    caps = np.linspace(0.1, 0.9, K)
    x = np.clip(caps[:, None] + rng.normal(0, .15, (K, nb)), 0, 1).mean(1)
    o = rng.normal(0, sd_o, K); g = rng.normal(0, sd_g, K)
    t = rng.random((K, n)) < touch  # paired: same touches both arms
    us = ((rng.random((K, n)) < np.clip(base+o+g, .02, .98)[:, None]) & t).sum(1) / t.sum(1)
    uf = ((rng.random((K, n)) < np.clip(base+o, .02, .98)[:, None]) & t).sum(1) / t.sum(1)
    return x, us, uf

def upper(x, y):
    r = stats.linregress(x, y); return r.slope + stats.t.ppf(.975, len(x)-2)*r.stderr

rng = np.random.default_rng(2026)
N = 4000
for K in (5, 6, 8):
    for sd_o, sd_g in ((0, 0), (0.1, 0.05), (0.1, 0.0)):
        mi = ms = 0
        for _ in range(N):
            x, us, uf = ladder(rng, K, sd_o=sd_o, sd_g=sd_g)
            mi += upper(x, us-uf) < 0; ms += upper(x, us) < 0
        se = np.sqrt(.025*.975/N)
        print(f"K={K} sd_o={sd_o} sd_g={sd_g}: P(I CI below 0)={mi/N:.4f}  P(static CI below 0)={ms/N:.4f}  (nominal 0.025, MC SE {se:.4f}, N={N})", flush=True)
