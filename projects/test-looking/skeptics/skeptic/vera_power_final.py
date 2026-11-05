# Vera final pre-data power table for PREREG v1 records (haiku .80, gem3.1 .30, gem3.5 .60, mini .75, nano .55), n=40/agent/arm.
# Rule v1 (s3): NO iff upper<1.08 first; YES iff lower>0 & upper>=1.08. Rule T (Tove #14861): YES iff lower>0 first; NO iff upper<0.73.
import numpy as np
from vera_power_band import mh, ex, lg
rng=np.random.default_rng(11056)
base=np.array([.80,.30,.60,.75,.55])
def pts(dd): return (base-ex(lg(base)-dd)).mean()*100
print("true cut (logits) | pts | v1: NO YES NS | v1@0.73 row: NO YES NS | Tove#14861: YES NO NS")
for dd in [0,0.18,0.36,0.55,0.73,1.08,1.4,-0.36]:
    R={'v1':[], 'v073':[], 'T':[]}
    for _ in range(4000):
        b=ex(lg(base)+rng.normal(0,0.5,5)); t=ex(lg(b)-dd)
        l,s=mh(rng.binomial(40,b),rng.binomial(40,t),40)
        lo,hi=l-1.96*s,l+1.96*s
        R['v1'].append('NO' if hi<1.08 else ('YES' if lo>0 else 'NS'))
        R['v073'].append('NO' if hi<0.73 else ('YES' if lo>0 else 'NS'))
        R['T'].append('YES' if lo>0 else ('NO' if hi<0.73 else 'NS'))
    f=lambda k,x: np.mean(np.array(R[k])==x)
    print(f"{dd:+.2f} | {pts(dd):5.1f} | {f('v1','NO'):.2f} {f('v1','YES'):.2f} {f('v1','NS'):.2f} | {f('v073','NO'):.2f} {f('v073','YES'):.2f} {f('v073','NS'):.2f} | {f('T','YES'):.2f} {f('T','NO'):.2f} {f('T','NS'):.2f}")
