# Power of the goal's yes-bar ("pre-switched T_B <= 0.5 x both controls, 3 of 3 seeds")
# under a true speed-up s, with per-run log-noise sd (parent T_dense sd_log 0.04-0.12 at k2-k5).
import numpy as np
rng=np.random.default_rng(0); N=200000
for sd in (0.05,0.10,0.15):
  for s in (1.0,1.5,1.94,2.0,2.5,3.0):
    pre=np.log(200/s)+sd*rng.standard_normal((N,3))
    c1=np.log(200)+sd*rng.standard_normal((N,2)); c2=np.log(200)+sd*rng.standard_normal((N,2))
    # bar A: vs mean of each control's seeds (log-mean); bar B: vs the min seed of each control
    mA=np.minimum(c1.mean(1),c2.mean(1)); mB=np.minimum(c1.min(1),c2.min(1))
    yA=(pre.max(1)<=mA+np.log(0.5)).mean(); yB=(pre.max(1)<=mB+np.log(0.5)).mean()
    print(f"sd {sd:.2f} speedup {s:.2f}: P(yes | vs control mean) {yA:.3f}  P(yes | vs control min seed) {yB:.3f}")
