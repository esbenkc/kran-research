# Law read (vs FRESH): held iff the seed-level 95% interval of R = T_MAIN/T_FRESH lies inside [lo, hi].
# P(held) under true R, for the proposed band [0.42, 0.63] and wider/narrower ones. Same noise model as vera_power_v2.py.
import numpy as np
from scipy import stats
rng=np.random.default_rng(3); N=100000
def held(R,sd,sdv,nF,band):
    S=5; v=sdv*rng.standard_normal((N,S)); main=np.log(296*R)+v+sd*rng.standard_normal((N,S))
    fresh=np.log(296)+v[:,:nF]+sd*rng.standard_normal((N,nF))
    d=main[:,:nF]-fresh if nF==S else np.concatenate([main[:,:nF]-fresh, main[:,nF:]-fresh.mean(1,keepdims=True)],1)
    m=d.mean(1); h=stats.t.ppf(.975,S-1)*d.std(1,ddof=1)/np.sqrt(S)
    return ((m-h>=np.log(band[0]))&(m+h<=np.log(band[1]))).mean()
for band in ((0.42,0.63),(0.40,0.66),(0.38,0.70)):
  for nF in (3,5):
    for sd,sdv in ((0.042,0.0),(0.08,0.10),(0.15,0.10)):
      row=" ".join(f"R{R}:{held(R,sd,sdv,nF,band):.3f}" for R in (0.3,0.4,0.515,0.6,0.75,1.0))
      print(f"band {band} nF {nF} sd {sd} sdv {sdv} | {row}")
