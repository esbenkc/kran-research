# Vera: rule NO iff MH CI upper < B; YES iff lower>0 and upper>=B; else NS. B in {0.73,1.08}; per-agent bases.
import numpy as np
from vera_power_band import mh, ex, lg
rng=np.random.default_rng(11054)
def run(base,B,n,reps=3000,drift=0.5):
    out={}
    for dd in [0,0.36,0.73,1.08]:
        r=[]
        for _ in range(reps):
            b=ex(lg(np.clip(base,0.02,0.98))+rng.normal(0,drift,len(base))); t=ex(lg(b)-dd)
            l,s=mh(rng.binomial(n,b),rng.binomial(n,t),n)
            if np.isnan(l): r.append('NS');continue
            lo,hi=l-1.96*s,l+1.96*s
            r.append('NO' if hi<B else ('YES' if lo>0 else 'NS'))
        r=np.array(r); out[dd]=(np.mean(r=='NO'),np.mean(r=='YES'),np.mean(r=='NS'))
    print(f"B={B} n={n} base={base}: "+"; ".join(f"shift {k}: NO {v[0]:.2f} YES {v[1]:.2f} NS {v[2]:.2f}" for k,v in out.items()))
for base in [[.05,.30,.95,.00,.55],[.90,.30,.60,.65,.55]]:
    for B in [0.73,1.08]:
        for n in [40,60]: run(np.array(base),B,n)
