# Vera power sim v2 (pre-data) for the split Tove/Kasper/Felix back:
# fit A{raw,bare} B{md,anchor}; held out A{md,anchor} B{raw,bare}. reach=1 assumed (CSS-blind formats).
# follow(A) from A-raw only (bare fit = reach only); follow(B) pooled over B-md + B-anchor.
# CONFIRMED-part: the 3 anchor-carrying held-out cells (A-md, A-anchor, B-raw) all |err|<=0.10 (point) and pred in [0.10,0.90].
import numpy as np
rng=np.random.default_rng(11232)
R=40000
def sim(n,pA,pB):
    fA=rng.binomial(n,pA,R)/n; fB=rng.binomial(2*n,pB,R)/(2*n)
    Amd=rng.binomial(n,pA,R)/n; Aan=rng.binomial(n,pA,R)/n; Braw=rng.binomial(n,pB,R)/n
    w=(np.abs(Amd-fA)<=0.1)&(np.abs(Aan-fA)<=0.1)&(np.abs(Braw-fB)<=0.1)
    inr=(fA>=0.1)&(fA<=0.9)&(fB>=0.1)&(fB<=0.9)
    return w.mean(),(w&inr).mean()
for n in (40,42,60):
  for pA,pB in ((0.5,0.5),(0.7,0.7),(0.8,0.8),(0.85,0.85),(0.93,0.8),(0.95,0.95)):
    a,b=sim(n,pA,pB)
    print(f"n={n} follow A={pA} B={pB}: P(3/3 within 10)={a:.2f}  and preds in [0.1,0.9]={b:.2f}")
