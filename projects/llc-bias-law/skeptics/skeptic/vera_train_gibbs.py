"""My Gibbs (own data draw, n=1024) on the TRAIN shapes at nbeta 1e3, q (k4=1/2) 0.1/1/10/100, vs Bruno's #30762 numbers."""
import json, numpy as np
from vera_core import *
out = {}
for r in (1, 8):
    M, N, H = 32, 32, 16
    lam, reg, lam0 = lam_parts(M, N, H, r)
    for seed in (0, 1):
        W0, B0, A0 = teacher(M, N, H, r, 10 + seed)
        X = np.random.default_rng(20 + seed).standard_normal((1024, M)); S = X.T @ X / 1024
        for q in (0.1, 1, 10, 100):
            g = np.sqrt(1000 * 0.5 / q)
            v = gibbs(W0, B0, A0, S, g, 1000.0, 1500, 300, np.random.default_rng(30 + seed))
            out.setdefault(f"r{r}_q{q}", []).append(v / lam - 1)
for k, v in out.items(): print(k, [round(100 * x, 1) for x in v])
json.dump(out, open("vera_train_gibbs.json", "w"), indent=1)
