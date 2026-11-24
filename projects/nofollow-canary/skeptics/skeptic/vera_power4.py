# Vera: alternative rule (pad leg only for status; body reported). Newcombe-Wilson 95%.
# C2 = G>=10pt AND (pad1-md) CI hi < 0 AND |pad1-raw| <= 10pt
# R2 = G>=10pt AND (pad1-raw) CI lo > 0      (length-matched prose stays above raw)
import numpy as np
rng=np.random.default_rng(3); S=20000; z=1.96
def wil(k,n):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def newc(k1,k2,n):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1,n); l2,u2=wil(k2,n)
    return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2), p1-p2+np.sqrt((u1-p1)**2+(p2-l2)**2)
def run(n,raw,md,pad):
    k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,pad=pad).items()}
    G=(k['md']-k['raw'])>=round(.10*n)
    _,hi=newc(k['pad'],k['md'],n); lo2,_=newc(k['pad'],k['raw'],n)
    C=G&(hi<0)&(abs(k['pad']-k['raw'])<=round(.10*n))
    R=G&(lo2>0)&~C
    return C.mean(),R.mean()
for raw,md in [(.74,.93),(.80,.93),(.65,.93)]:
    g=md-raw
    for n in (60,90):
        out=[f"{nm} C {c:.3f} R {r:.3f}" for nm,pad in [('length',raw),('markup',md),('half',raw+g/2),('3/4 length',raw+g/4)] for c,r in [run(n,raw,md,pad)]]
        print(f"raw {raw} md {md} n {n}: "+" | ".join(out))
print("no gap (.90 all) n90:", run(90,.9,.9,.9))
