# Vera: Felix FINAL PLAN v1 (#29570) exact rule at n=60, counts. Compare REJECTED forms.
# G: md-raw >= 6. CONF: |pad1-raw|<=6 AND |body-md|<=6.
# R_kasper: pad1-md Newcombe lo > -0.10 ; R_ifnot: |pad1-md|<=3 AND md-body>=9 ; R_vera: pad1-raw Newcombe lo > 0
import numpy as np
rng=np.random.default_rng(4); S=40000; z=1.96; n=60
def wil(k,n):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def newc(k1,k2,n):
    p1,p2=k1/n,k2/n; l1,u1=wil(k1,n); l2,u2=wil(k2,n)
    return p1-p2-np.sqrt((p1-l1)**2+(u2-p2)**2), p1-p2+np.sqrt((u1-p1)**2+(p2-l2)**2)
for raw,md in [(.74,.93),(.80,.93)]:
    g=md-raw
    sc={'length token-linear (body=raw+.23G)':(raw,raw+.23*g),'length naive (body=md)':(raw,md),
        'markup (pad=md, body=raw)':(md,raw),'half':(raw+g/2,raw+g/2)}
    for name,(pad,body) in sc.items():
        k={a:rng.binomial(n,v,S) for a,v in dict(raw=raw,md=md,pad=pad,body=body).items()}
        G=(k['md']-k['raw'])>=6
        C=G&(abs(k['pad']-k['raw'])<=6)&(abs(k['body']-k['md'])<=6)
        out=[f"CONF {C.mean():.3f}"]
        for rn,R in [('R_kasper',newc(k['pad'],k['md'],n)[0]>-.10),('R_ifnot',(abs(k['pad']-k['md'])<=3)&((k['md']-k['body'])>=9)),('R_vera',newc(k['pad'],k['raw'],n)[0]>0)]:
            out.append(f"{rn} {(G&R&~C).mean():.3f}")
        print(f"raw {raw} md {md} {name:36s} "+"  ".join(out))
