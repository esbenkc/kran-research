import numpy as np
from scipy import stats
s=0.0316; rng=np.random.default_rng(1)
for n in [3,5]:
  t=stats.t.ppf(.975,n-1)
  for k in [1,2,3]:
    x=rng.normal(-0.922,k*s,(200000,n)); mm=x.mean(1); h=t*x.std(1,ddof=1)/np.sqrt(n)
    print(n,k,round(np.mean((mm-h>=-1.4)&(mm+h<=-0.7)),3))
