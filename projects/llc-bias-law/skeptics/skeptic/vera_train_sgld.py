"""My own minibatch SGLD (vera_core.sgld, 4 chains, own data/teacher draws) on TRAIN cells with the frozen T map
(timing_S.json), vs my Gibbs; gives the sampler-bias floor I expect in Bruno's J2 rows."""
import json, sys, time, numpy as np
from vera_core import *
tm = json.load(open('/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law/data/timing_S.json'))['settings']
out = []
t0 = time.time()
for r in (1, 8):
    M, N, H = 32, 32, 16
    lam, reg, lam0 = lam_parts(M, N, H, r)
    W0, B0, A0 = teacher(M, N, H, r, 41)
    X = np.random.default_rng(42).standard_normal((1024, M)); S = X.T @ X / 1024
    for nb, q, c in ((1000, 1, 0.01), (1000, 1, 0.1), (100, 10, 0.03), (10000, 0.1, 0.1), (1000, 100, 0.1), (100, 0.1, 0.1)):
        key = f"nb{nb}_q{q}_c{c}"; T = tm[key]['T']
        g = np.sqrt(nb * 0.5 / q); eps = c / (nb * 2.0 + g)
        sg = sgld(W0, B0, A0, X, g, float(nb), eps, T, 64, np.random.default_rng(43), burn_frac=0.5, chains=4)
        gb = gibbs(W0, B0, A0, S, g, float(nb), 1200, 240, np.random.default_rng(44))
        o = dict(r=r, key=key, T=T, sgld=float(sgld_mean := sg.mean() / lam - 1), sgld_chains=(sg / lam - 1).tolist(), gibbs=gb / lam - 1)
        out.append(o)
        print(f"r{r} {key:22s} T={T:7d} SGLD {100*o['sgld']:+6.1f} (chains {[round(100*x,1) for x in o['sgld_chains']]}) Gibbs {100*o['gibbs']:+6.1f} gap {100*(o['sgld']-o['gibbs']):+5.1f} [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open('vera_train_sgld.json', 'w'), indent=1)
