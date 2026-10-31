"""Pre-data power sim of the frozen test: stronger five pooled, U-R diff, 95% CI excl 0, U>=0.40, R<=0.20.
Worlds: per-model logit-normal heterogeneity around cell means. n=20 runs/cell, K=5 models.
CIs: (a) run-level bootstrap pooled, (b) model-cluster bootstrap, (c) model-level t (4 df)."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
K, N, B, W = 5, 20, 1000, 400
def lg(p): p=np.clip(p,1e-3,1-1e-3); return np.log(p/(1-p))
def sig(x): return 1/(1+np.exp(-x))
def effects(c):  # c[..., 4] cells: UR, UN, LR, LN  (U=useful, L=useless, R=real, N=not real)
    U = (c[...,0]+c[...,1])/2 - (c[...,2]+c[...,3])/2
    R = (c[...,0]+c[...,2])/2 - (c[...,1]+c[...,3])/2
    return U, R
worlds = {
 "prereg-true (UR.9 UN.8 LR.25 LN.1)": [.9,.8,.25,.1],
 "realism-driven (UR.9 UN.3 LR.7 LN.1)": [.9,.3,.7,.1],
 "both (UR.9 UN.55 LR.45 LN.1)": [.9,.55,.45,.1],
 "null-diff (UR.9 UN.5 LR.5 LN.1)": [.9,.5,.5,.1],
}
for sd in [0.5, 1.0]:
  for name, m in worlds.items():
    hits = {"run":[0,0], "clus":[0,0], "t":[0,0]}
    for w in range(W):
        p = sig(lg(np.array(m))[None,:] + rng.normal(0, sd, (K,1)) + rng.normal(0, sd/2, (K,4)))
        y = rng.binomial(1, p[...,None].repeat(N,-1))  # K,4,N
        rates = y.mean(-1)
        U, R = effects(rates); d = U - R
        D, Um, Rm = d.mean(), U.mean(), R.mean()
        # run-level bootstrap: resample runs within each (model, cell)
        idx = rng.integers(0, N, (B, K, 4, N))
        rb = np.take_along_axis(y[None].repeat(B,0), idx, -1).mean(-1)
        Ub, Rb = effects(rb); db = (Ub-Rb).mean(-1)
        lo, hi = np.percentile(db, [2.5, 97.5])
        # cluster bootstrap over models
        mi = rng.integers(0, K, (B, K)); dc = d[mi].mean(-1)
        clo, chi = np.percentile(dc, [2.5, 97.5])
        h = stats.t.ppf(.975, K-1) * d.std(ddof=1)/np.sqrt(K)
        base = Um >= .40 and Rm <= .20
        for k,(l,_) in {"run":(lo,hi),"clus":(clo,chi),"t":(D-h,D+h)}.items():
            hits[k][0] += (l > 0); hits[k][1] += (l > 0) and base
    print(f"sd={sd} {name}: " + " ".join(f"{k}: CIexcl0 {v[0]/W:.2f} full-prereg {v[1]/W:.2f}" for k,v in hits.items()))
