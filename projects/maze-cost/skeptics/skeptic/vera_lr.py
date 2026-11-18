"""Vera: power of the frozen constancy LR test (decisions at depths 1-3 vs 4..), K=9 decisions, n=40."""
import numpy as np
from scipy.stats import chi2
from vera_law_power import sim_depths
from vera_mean import shapes
K=9
def lr(dep):
    # depth D_r: leave at D (D<=K... sim: D==K means cap reached -> censored at K with K-1? sim gives D<=K; treat D==K and no leave as censored)
    L=[0,0];E=[0,0]
    for D in dep:
        left = D < K
        dec = D if left else K  # sim: K decisions max, all go-on if not left
        for d in range(1,dec+1):
            g=0 if d<=3 else 1; E[g]+=1
            if left and d==D: L[g]+=1
    def ll(l,e):
        p=l/e if e else 0
        return (l*np.log(p) if l else 0)+((e-l)*np.log(1-p) if e-l else 0)
    if E[1]==0: return None
    full=ll(L[0],E[0])+ll(L[1],E[1]); red=ll(L[0]+L[1],E[0]+E[1])
    return chi2.sf(2*(full-red),1)
rng=np.random.default_rng(3)
for name,fac in shapes:
    ps=[lr(sim_depths(fac(rng,40),40,K,rng)) for _ in range(1000)]
    ok=[p for p in ps if p is not None]
    print(f"{name:20s} reject@0.05={np.mean([p<0.05 for p in ok]):.2f} (defined {len(ok)/len(ps):.2f})")
