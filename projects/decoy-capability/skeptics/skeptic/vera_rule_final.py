"""Final s6 candidate (Tove #9008/#9009, Kasper #9007): t-intervals K-2 df.
CONFIRMED = CI(I) and CI(static change) entirely below 0 at level Lc, AND static point <= -0.10.
REJECTED  = 90% CI(static change) entirely above -0.15 OR 90% CI(I) entirely above -0.15. Both -> REJECTED.
Design: K rungs, 30 paired seeds, 15 nobait eps, touch p, model offset sd 0.10, gap noise sd 0.05."""
import numpy as np
from scipy import stats

def ladder(rng, K, drop, n=30, nb=15, sd_o=0.1, sd_g=0.05, touch=0.6, base=0.6):
    caps = np.linspace(0.1, 0.9, K)
    x = np.clip(caps[:, None] + rng.normal(0, .15, (K, nb)), 0, 1).mean(1)
    f = (caps - caps[0]) / (caps[-1] - caps[0])
    o = rng.normal(0, sd_o, K); g = rng.normal(0, sd_g, K)
    t = rng.random((K, n)) < touch
    nt = np.maximum(t.sum(1), 1)
    us = ((rng.random((K, n)) < np.clip(base+o+g-drop*f, .02, .98)[:, None]) & t).sum(1) / nt
    uf = ((rng.random((K, n)) < np.clip(base+o, .02, .98)[:, None]) & t).sum(1) / nt
    return x, us, uf, t.sum(1)

def ci(x, y, R, lvl):
    r = stats.linregress(x, y); h = stats.t.ppf(1-(1-lvl)/2, len(x)-2)*r.stderr
    return np.array([r.slope-h, r.slope+h])*R, r.slope*R

def status(x, us, uf, Lc):
    R = x.max()-x.min()
    Ic, _ = ci(x, us-uf, R, Lc); Sc, sp = ci(x, us, R, Lc)
    I9, _ = ci(x, us-uf, R, .90); S9, _ = ci(x, us, R, .90)
    if S9[0] > -0.15 or I9[0] > -0.15: return "R"
    if Ic[1] < 0 and Sc[1] < 0 and sp <= -0.10: return "C"
    return "U"

rng = np.random.default_rng(1028)
print("K drop touch | Lc=0.90: C R U | Lc=0.95: C R U | N")
for K in (6, 8, 10, 12):
    for drop in (0.0, 0.2, 0.3):
        for touch in (0.6, 0.4):
            N = 3000 if drop == 0 else 1500
            c = {.90: dict(C=0, R=0, U=0), .95: dict(C=0, R=0, U=0)}
            for _ in range(N):
                x, us, uf, nt = ladder(rng, K, drop, touch=touch)
                for L in c: c[L][status(x, us, uf, L)] += 1
            f = lambda d: " ".join(f"{d[k]/N:.3f}" for k in "CRU")
            print(K, drop, touch, "|", f(c[.90]), "|", f(c[.95]), "|", N, flush=True)
