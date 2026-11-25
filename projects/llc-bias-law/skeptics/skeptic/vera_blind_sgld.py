"""Independent sampler check on BLIND cells (run after predictions.json push 8c50d45be, 06:31:11 PST).
Own teacher/data draws, own SGLD (4 chains) and own Gibbs, frozen T map; 3 settings x 5 blind shapes (plan #30744)."""
import json, time, numpy as np
from vera_core import *
tm = json.load(open('/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law/data/timing_S.json'))['settings']
BLIND = [(24, 24, 12, 3), (24, 24, 12, 6), (48, 48, 8, 2), (20, 20, 10, 5), (40, 24, 12, 4)]
SET = [(1000, 1, 0.01), (1000, 1, 0.1), (10000, 10, 0.03)]
out = []; t0 = time.time()
for (M, N, H, r) in BLIND:
    lam, reg, lam0 = lam_parts(M, N, H, r)
    W0, B0, A0 = teacher(M, N, H, r, 500 + r)
    X = np.random.default_rng(600 + r).standard_normal((1024, M)); S = X.T @ X / 1024
    for nb, q, c in SET:
        key = f"nb{nb}_q{q}_c{c}"; T = tm[key]['T']
        g = np.sqrt(nb * 0.5 / q); eps = c / (nb * 2.0 + g)
        sg = sgld(W0, B0, A0, X, g, float(nb), eps, T, 64, np.random.default_rng(700 + r), burn_frac=0.5, chains=4)
        gb = gibbs(W0, B0, A0, S, g, float(nb), 1200, 240, np.random.default_rng(800 + r))
        o = dict(shape=[M, N, H, r], key=key, T=T, e_sgld=float(sg.mean() / lam - 1), e_chains=(sg / lam - 1).tolist(), e_gibbs=float(gb / lam - 1))
        out.append(o)
        print(f"{M}x{N}x{H}r{r} {key:20s} T={T:6d} SGLD {100*o['e_sgld']:+6.1f} Gibbs {100*o['e_gibbs']:+6.1f} [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open('vera_blind_sgld.json', 'w'), indent=1)
