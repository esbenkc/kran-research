# Vera power sim (pre-data): true product law; held-out cell catch vs pred from fit cells.
import numpy as np
rng=np.random.default_rng(1123)
R=20000
for n in (40,60,80):
  for p in (0.3,0.5,0.7,0.9):
    nfit=2*n  # follow pooled over 2 fit cells of same model
    obs=rng.binomial(n,p,(R,4))/n
    pred=rng.binomial(nfit,p,(R,4))/nfit
    ok=(np.abs(obs-pred)<=0.10).sum(1)
    print(f"n={n} p={p}: P(cell within 10)={np.mean(np.abs(obs-pred)<=0.10):.2f}  P(>=3 of 4)={np.mean(ok>=3):.2f}  P(4 of 4)={np.mean(ok==4):.2f}")
# false-miss: bare cell truly at gap d below pred -> P(miss >= 15)
for n in (40,60):
  for d in (0.15,0.20,0.30):
    p=0.8; obs=rng.binomial(n,p-d,R)/n; pred=rng.binomial(2*n,p,R)/(2*n)
    print(f"n={n} true gap {d}: P(observed miss>=0.15)={np.mean(pred-obs>=0.15):.2f}")
