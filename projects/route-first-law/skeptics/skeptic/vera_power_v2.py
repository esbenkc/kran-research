# Power sim of Felix's DECISION rule (#31162), before the freeze.
# 4 cells (2 harness x 2 models). Per cell arms NONE, CB (correct-before), CA (correct-after), n per arm = 3 blocks x B.
# CONFIRMED: every cell CB point <= .25 and CA point >= .75 (cells valid: NONE point >= .75).
# REJECTED: some valid cell has CB Wilson low > .25 or CA Wilson high < .75. Else UNCONFIRMED.
# Seed-block overdispersion: block rate ~ Beta with mean p, concentration kappa.
import numpy as np, sys
rng=np.random.default_rng(1126)
Z=1.96
def wil(k,n):
    p=k/n; d=1+Z*Z/n; c=(p+Z*Z/(2*n))/d; h=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d; return c-h,c+h
def draw(p,B,kappa,size):
    if kappa is None: return rng.binomial(3*B,p,size)
    a=max(p*kappa,1e-3); b=max((1-p)*kappa,1e-3)
    q=rng.beta(a,b,(size,3)); return rng.binomial(B,q).sum(1)
def sim(none,cb,ca,B=10,kappa=None,N=20000,ncell=4,tC=.25,tA=.75):
    n=3*B; out={'CONF':0,'REJ':0,'UNC':0}
    K=[(draw(none,B,kappa,N),draw(cb,B,kappa,N),draw(ca,B,kappa,N)) for _ in range(ncell)]
    conf=np.ones(N,bool); rej=np.zeros(N,bool)
    for kn,kb,ka in K:
        valid=kn/n>=.75
        conf&= valid & (kb/n<=tC) & (ka/n>=tA)
        lb,_=wil(kb,n); _,ua=wil(ka,n)
        rej|= valid & ((lb>tC)|(ua<tA))
    c=conf.mean(); r=(rej&~conf).mean()
    return c,r,1-c-r
scen={'law true 15/85':(.95,.15,.85),'law true 20/80':(.95,.20,.80),'law strong 10/90':(.95,.10,.90),
      'presence only 15/15':(.95,.15,.15),'half order 15/50':(.95,.15,.50),'no route effect':(.95,.95,.95),
      'CA mild 15/70':(.95,.15,.70)}
for B in (10,20):
  for kappa in (None,20):
    print(f'--- n/arm={3*B} kappa={kappa}')
    for k,v in scen.items():
        c,r,u=sim(*v,B=B,kappa=kappa); print(f'{k:22s} CONF {c:.3f} REJ {r:.3f} UNC {u:.3f}')
