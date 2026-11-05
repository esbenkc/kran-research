# Vera: Tove #14861 precedence. YES iff MH CI lower>0 (first); NO iff lower<=0 and upper<B (B=0.73); else NS.
# Bases: Iris #14858 frozen picks haiku .80, gem3.1 .30, gem3.5 .60, nano .55, mini .65. Drift sd 0.5 logit on each base.
import numpy as np
from vera_power_band import mh, ex, lg
rng=np.random.default_rng(11055)
def run(base,B,n,reps=4000,drift=0.5,shifts=(0,0.18,0.36,0.73,1.08,-0.36)):
    print(f"B={B} n={n}/agent/arm base={list(base)} drift_sd={drift}")
    for dd in shifts:
        r=[];hw=[]
        for _ in range(reps):
            b=ex(lg(np.clip(base,0.02,0.98))+rng.normal(0,drift,len(base))); t=ex(lg(b)-dd)
            l,s=mh(rng.binomial(n,b),rng.binomial(n,t),n)
            if np.isnan(l): r.append('NS');continue
            lo,hi=l-1.96*s,l+1.96*s; hw.append(1.96*s)
            r.append('YES' if lo>0 else ('NO' if hi<B else 'NS'))
        r=np.array(r)
        print(f"  true cut {dd:+.2f} logits: YES {np.mean(r=='YES'):.3f} NO {np.mean(r=='NO'):.3f} NS {np.mean(r=='NS'):.3f}  CI half-width {np.mean(hw):.2f}")
base=np.array([.80,.30,.60,.55,.65])
for n in (40,60): run(base,0.73,n)
run(base,0.73,60,drift=1.0)
run(np.array([.05,.30,.95,.00,.55]),0.73,60)
