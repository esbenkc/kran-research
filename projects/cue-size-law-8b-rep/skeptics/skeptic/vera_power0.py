# P(CONFIRMED-interval) for n=3 fresh seeds, given parent per-seed slopes (seen data)
import numpy as np
from scipy import stats
par=np.array([-0.8863,-0.8898,-0.9362,-0.940,-0.956])
m,s=par.mean(),par.std(ddof=1)
rng=np.random.default_rng(0); t=stats.t.ppf(.975,2)
for mu in [m,-0.80,-0.75,-1.0]:
  for sd in [s,2*s,3*s,5*s]:
    x=rng.normal(mu,sd,(200000,3)); mm=x.mean(1); h=t*x.std(1,ddof=1)/np.sqrt(3)
    p=np.mean((mm-h>=-1.4)&(mm+h<=-0.7)); r=np.mean((mm+h<-1.4)|(mm-h>-0.7))
    print(f"mu {mu:+.3f} sd {sd:.3f}: P(inside) {p:.3f}  P(outside) {r:.4f}")
