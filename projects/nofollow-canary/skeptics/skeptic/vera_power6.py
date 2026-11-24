# Vera: can the CONFIRMED form push P(CONF | half) below 0.15 while keeping P(CONF | length)? n_status=90.
import numpy as np
rng=np.random.default_rng(5); S=40000; z=1.96
def wil(k,n):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def newc(k1,k2,n):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1,n); l2,u2=wil(k2,n)
    return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2), p1-p2+np.sqrt((u1-p1)**2+(p2-l2)**2)
for n in (90,60):
  for raw,md in [(.74,.93),(.80,.93)]:
    g=md-raw; row=[]
    for nm,pad in [('length',raw),('3/4',raw+g/4),('half',raw+g/2),('markup',md)]:
        k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,pad=pad).items()}
        G=(k['md']-k['raw'])>=round(.1*n); _,h1=newc(k['pad'],k['md'],n); l2,h2=newc(k['pad'],k['raw'],n)
        a=G&(h1<0)&(abs(k['pad']-k['raw'])<=round(.1*n))
        b=G&(h1<0)&(h2<.10)
        dg=np.maximum(k['md']-k['raw'],1); c=G&(h1<0)&((k['md']-k['pad'])/dg>=.75)
        row.append(f"{nm}: a {a.mean():.2f} b {b.mean():.2f} c {c.mean():.2f}")
    print(f"n {n} raw {raw}: "+" | ".join(row))
