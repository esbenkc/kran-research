# Vera: REJ alternatives under the final v2 CONF. REJ1 = frozen draft. REJ2 = G & lo(md-rawnf)>0 & (lo(mdnf-raw)>0 | mdnf >= mdhtml-9)
import numpy as np
rng=np.random.default_rng(2412); S=50000; z=1.96; n=90
def wil(k):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def lo(k1,k2):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1); l2,u2=wil(k2); return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2)
for raw,md in [(.74,.93),(.80,.93)]:
    g=md-raw
    sc={'nofollow true':(md,raw,md),'nofollow null':(raw,md,md),'half nofollow':(raw+g/2,md-g/2,md),
        'raw-side only':(md,md,md),'nofollow raw + syntax md':(md,raw,raw),'syntax only':(raw,raw,raw),'md-side only':(raw,raw,md)}
    for nm,(rnf,mnf,mh) in sc.items():
        k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,rnf=rnf,mnf=mnf,mh=mh).items()}
        G=(k['md']-k['raw'])>=9
        C=G&(lo(k['rnf'],k['raw'])>0)&(lo(k['mh'],k['mnf'])>0)&(k['rnf']>=k['md']-9)&(k['mnf']<=k['raw']+9)
        R1=G&(lo(k['md'],k['rnf'])>0)&(lo(k['mnf'],k['raw'])>0)
        R2=G&(lo(k['md'],k['rnf'])>0)&((lo(k['mnf'],k['raw'])>0)|(k['mnf']>=k['mh']-9))
        ov=(C&R2).mean(); R2=R2&~C
        print(f"raw {raw:.2f} {nm:26s} CONF {C.mean():.3f}  REJ1 {R1.mean():.3f}  REJ2 {R2.mean():.3f} (overlap with CONF {ov:.4f})")
