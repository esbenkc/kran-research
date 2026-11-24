# Vera: Felix's draft rule (#29563 + Kasper #29567), Newcombe-Wilson 95% CIs, n per status arm 60 or 90.
# CONFIRMED = G>=10pt AND (pad1-md) CI hi < 0 AND |pad1-raw|<=10pt AND |body-md|<=10pt
# REJECTED  = G>=10pt AND (pad1-md) CI lo > -10pt ; else UNCONFIRMED.
import numpy as np
rng=np.random.default_rng(2); S=20000; z=1.96
def wil(k,n):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def newc(k1,k2,n):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1,n); l2,u2=wil(k2,n)
    return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2), p1-p2+np.sqrt((u1-p1)**2+(p2-l2)**2)
def run(n,raw,md,pad,body):
    k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,pad=pad,body=body).items()}
    G=(k['md']-k['raw'])/n>=.10-1e-9
    lo,hi=newc(k['pad'],k['md'],n)
    C=G&(hi<0)&(abs(k['pad']-k['raw'])/n<=.10+1e-9)&(abs(k['body']-k['md'])/n<=.10+1e-9)
    R=G&(lo>-.10)&~C
    return C.mean(),R.mean(),1-C.mean()-R.mean()
for raw,md in [(.74,.93),(.80,.93)]:
    g=md-raw
    sc={'length, body=raw+0.23G (token-linear)':(raw,raw+.23*g),
        'length, body=md (naive)':(raw,md),
        'markup (pad=md, body=raw)':(md,raw),
        'half':(raw+g/2,raw+g/2)}
    for n in (60,90):
        for name,(pad,body) in sc.items():
            c,r,u=run(n,raw,md,pad,body); print(f"raw {raw} md {md} n {n} {name:40s} CONF {c:.3f} REJ {r:.3f} UNC {u:.3f}")
c,r,u=run(90,.90,.90,.90,.90); print("no gap at all (all .90) n 90: CONF %.3f REJ %.3f"%(c,r))
