"""Vera: outcome probabilities under the FROZEN window (v1.1): K=12 maze pages after entry,
TESTED iff fitted d* > 3 and fitted (1-h)^K < 0.5; pass iff pred/obs median spend in [0.5,2]."""
import numpy as np
from vera_law_power import sim_depths, cost
import sys; K=int(sys.argv[1])
def rule(dep):
    views=np.minimum(dep,3).sum(); leaves=((dep<=3)&(dep<K)).sum(); h=leaves/views
    if h<=0 or (1-h)**K>=0.5: return "never_quits"
    d=int(np.ceil(np.log(0.5)/np.log(1-h))) if h<1 else 1
    if d<=3: return "quits_in_pilot"
    r=cost(d)/np.median(cost(dep)); return "pass" if 0.5<=r<=2 else "fail"
rng=np.random.default_rng(1)
print("band of h that is TESTED:", [round(h,3) for h in np.arange(0.01,0.4,0.005) if (1-h)**K<0.5 and np.ceil(np.log(.5)/np.log(1-h))>3][::4])
for n in (40,24):
  for name,f in [("geo .03",lambda d,r:.03),("geo .08",lambda d,r:.08),("geo .12",lambda d,r:.12),("geo .18",lambda d,r:.18),("geo .25",lambda d,r:.25),
                 ("rising .03d",lambda d,r:min(1,.03*d)),("falling .4/d",lambda d,r:.4/d)]:
    c={}
    for _ in range(2000):
        o=rule(sim_depths(f,n,K,rng)); c[o]=c.get(o,0)+1
    print(f"n={n} {name:13s}", {k:round(v/2000,2) for k,v in sorted(c.items())})
