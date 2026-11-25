"""Independent check of the post's 'perfect sampler on noise-free data reads 30% to 33% low' at the parent's spring
(gamma 10, nbeta 128/ln128 = 26.38) on the training shapes (32,32,16) r1 and r8; own Gibbs, own draws, 2 seeds."""
import json, numpy as np
from vera_core import *
nb = 128 / np.log(128); out = {}
for r in (1, 8):
    lam = lam_parts(32, 32, 16, r)[0]; v = []
    for s in (0, 1):
        W0, B0, A0 = teacher(32, 32, 16, r, 900 + s)
        X = np.random.default_rng(910 + s).standard_normal((1024, 32)); S = X.T @ X / 1024
        v.append(gibbs(W0, B0, A0, S, 10.0, nb, 2000, 400, np.random.default_rng(920 + s)) / lam - 1)
    out[f"r{r}"] = v; print(f"r{r}: nbeta {nb:.2f} gamma 10 -> settled e = {[round(100*x,1) for x in v]} %")
json.dump(out, open('vera_s2b.json', 'w'))
