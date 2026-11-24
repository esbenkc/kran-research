# Vera: v2 nofollow rule (Felix #29584), n=90 per arm, counts, Newcombe-Wilson 95%.
import numpy as np
rng=np.random.default_rng(7); S=40000; z=1.96; n=90
def wil(k,n):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def newc(k1,k2,n):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1,n); l2,u2=wil(k2,n)
    return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2), p1-p2+np.sqrt((u1-p1)**2+(p2-l2)**2)
lo=lambda a,b: newc(a,b,n)[0]
for raw,md in [(.74,.93),(.80,.93)]:
    g=md-raw
    sc={'nofollow true':(md,raw),'nofollow null':(raw,md),'half':(raw+g/2,md-g/2),
        'raw-side only (rawnf=md, mdnf=md)':(md,md),'md-side only (rawnf=raw, mdnf=raw)':(raw,raw),'3/4':(raw+.75*g,md-.75*g)}
    for nm,(rnf,mnf) in sc.items():
        k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,rnf=rnf,mnf=mnf).items()}
        G=(k['md']-k['raw'])>=9
        ci=G&(lo(k['rnf'],k['raw'])>0)&(lo(k['md'],k['mnf'])>0)
        C=ci&(k['rnf']>=k['md']-9)&(k['mnf']<=k['raw']+9)
        R=G&(lo(k['md'],k['rnf'])>0)&(lo(k['mnf'],k['raw'])>0)
        print(f"raw {raw} {nm:36s} CONF {C.mean():.3f}  CONF(CI legs only) {ci.mean():.3f}  REJ {R.mean():.3f}")
