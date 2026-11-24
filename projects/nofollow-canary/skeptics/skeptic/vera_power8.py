# Vera: Kasper #29585 leg swap: second CONF leg (mdhtml - mdnf) lo>0 vs (md - mdnf) lo>0. n=90 all arms.
import numpy as np
rng=np.random.default_rng(8); S=40000; z=1.96; n=90
def wil(k,n):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def lo(k1,k2):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1,n); l2,u2=wil(k2,n); return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2)
raw,md=.74,.93
sc={'nofollow true, no syntax':(md,raw,md),'nofollow in raw + syntax in md':(md,raw,raw),
    'nofollow true + half syntax':(md,raw,(raw+md)/2),'null':(raw,md,md),'syntax only':(raw,raw,raw)}
for nm,(rnf,mnf,mh) in sc.items():
    k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,rnf=rnf,mnf=mnf,mh=mh).items()}
    G=(k['md']-k['raw'])>=9; base=G&(lo(k['rnf'],k['raw'])>0)&(k['rnf']>=k['md']-9)&(k['mnf']<=k['raw']+9)
    old=base&(lo(k['md'],k['mnf'])>0); new=base&(lo(k['mh'],k['mnf'])>0)
    print(f"{nm:34s} CONF old {old.mean():.3f}  CONF kasper {new.mean():.3f}")
