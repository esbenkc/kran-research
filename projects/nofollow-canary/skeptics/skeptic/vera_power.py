# Vera: P(rule says YES) under true scenarios, n per arm. Rule variants from the goal text (pre-freeze).
import numpy as np
rng=np.random.default_rng(0); N=60; S=20000
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def diffci(k1,k2,n):  # newcombe-ish normal approx for p1-p2
    p1,p2=k1/n,k2/n; se=np.sqrt((p1*(1-p1)+p2*(1-p2))/n+1e-9); return p1-p2-1.96*se,p1-p2+1.96*se
sc={'length':dict(raw=.74,pad=.74,body=.93,md=.93),
    'markup':dict(raw=.74,pad=.93,body=.74,md=.93),
    'half':dict(raw=.74,pad=.835,body=.835,md=.93),
    'none(both equal .74 except md)':dict(raw=.74,pad=.93,body=.93,md=.93)}
PRED=0.964
for name,p in sc.items():
    k={a:rng.binomial(N,v,S) for a,v in p.items()}
    pt=(abs(k['pad']-k['raw'])/N<=.10)&(abs(k['body']-k['md'])/N<=.10)
    lo1,hi1=diffci(k['pad'],k['raw'],N); lo2,hi2=diffci(k['body'],k['md'],N)
    eq=(lo1>=-.10)&(hi1<=.10)&(lo2>=-.10)&(hi2<=.10)
    lo,hi=wilson(k['raw'],N); law=(hi-PRED)< -.10
    print(f"{name:32s} point-within10 {pt.mean():.3f}  +law {(pt&law).mean():.3f}  CI-equivalence {eq.mean():.3f}  law alone {law.mean():.3f}")
