# Vera pre-data sim of Tove's two-sided band: stratified (agent) Mantel-Haenszel log OR, 95% CI (Robins-Breslow-Greenland),
# CONFIRMED-NO if CI inside +-B logits, YES if CI lower > 0 (drop) and not inside band; else NS. n=40/agent/arm.
import numpy as np
rng=np.random.default_rng(11052)
lg=lambda p: np.log(p/(1-p)); ex=lambda x:1/(1+np.exp(-x))
def mh(xp,xt,n):
    a=xp; b=n-xp; c=xt; d=n-xt; N=2*n
    R=a*d/N; S=b*c/N
    if R.sum()==0 or S.sum()==0: return np.nan,np.nan
    P=(a+d)/N; Q=(b+c)/N
    orr=R.sum()/S.sum()
    v=(P*R).sum()/(2*R.sum()**2)+((P*S+Q*R)).sum()/(2*R.sum()*S.sum())+(Q*S).sum()/(2*S.sum()**2)
    return np.log(orr), np.sqrt(v)
def run(base,dlist,B,n=40,reps=3000,drift=0.5):
    print(f"base={base} band=+-{B} n={n}/agent/arm")
    for dd in dlist:
        res=[]
        for _ in range(reps):
            b=ex(lg(np.clip(base,0.02,0.98))+rng.normal(0,drift,len(base)))
            t=ex(lg(b)-dd)
            l,s=mh(rng.binomial(n,b),rng.binomial(n,t),n)
            if np.isnan(l): res.append('NS'); continue
            lo,hi=l-1.96*s,l+1.96*s
            if lo>-B and hi<B: res.append('NO')
            elif lo>0: res.append('YES')
            else: res.append('NS')
        res=np.array(res)
        print(f"  true shift {dd:4.2f} logits: P(NO)={np.mean(res=='NO'):.2f} P(YES)={np.mean(res=='YES'):.2f} P(NS)={np.mean(res=='NS'):.2f}")
D=[0,0.3,0.6,1.08,1.6]
run(np.array([0.05,0.30,0.95,0.00,0.55]),D,1.08)
run(np.array([0.40,0.45,0.55,0.35,0.50]),D,1.08)
run(np.array([0.40,0.45,0.55,0.35,0.50]),D,1.08,n=60)
