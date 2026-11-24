# Vera: FINAL PLAN v2 (#29590) frozen rule, exact form, n=90 per status arm, counts, Newcombe-Wilson 95%.
# G = md-raw >= 9. CONF = G & lo(rawnf-raw)>0 & lo(mdhtml-mdnf)>0 & rawnf >= md-9 & mdnf <= raw+9.
# REJ = G & lo(md-rawnf)>0 & lo(mdnf-raw)>0.
import numpy as np
rng=np.random.default_rng(2411); S=50000; z=1.96; n=90
def wil(k):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def lo(k1,k2):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1); l2,u2=wil(k2); return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2)
def rule(k):
    G=(k['md']-k['raw'])>=9
    C=G&(lo(k['rnf'],k['raw'])>0)&(lo(k['mh'],k['mnf'])>0)&(k['rnf']>=k['md']-9)&(k['mnf']<=k['raw']+9)
    R=G&(lo(k['md'],k['rnf'])>0)&(lo(k['mnf'],k['raw'])>0)
    assert not (C&R).any(); return C.mean(),R.mean()
for raw,md in [(.74,.93),(.80,.93),(.65,.93)]:
    g=md-raw
    sc={'nofollow true (no syntax)':(md,raw,md),'nofollow null':(raw,md,md),'half nofollow':(raw+g/2,md-g/2,md),
        'raw-side only':(md,md,md),'nofollow raw + syntax md':(md,raw,raw),'syntax only':(raw,raw,raw)}
    for nm,(rnf,mnf,mh) in sc.items():
        k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,rnf=rnf,mnf=mnf,mh=mh).items()}
        c,r=rule(k); print(f"raw {raw:.2f} md {md:.2f} {nm:28s} P(CONF) {c:.3f}  P(REJ) {r:.3f}  P(UNC) {1-c-r:.3f}")
k={a:rng.binomial(n,.9,S) for a in ['raw','md','rnf','mnf','mh']}; print("no gap (all .90):",rule(k))
