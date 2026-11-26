# Power sim of Tove's change (3): clauses pool the 2 Gemma sizes per harness -> 2 cells x (C1,C2).
# Each pooled arm = 2 models x 3 seed blocks x B runs. Model-level rates may differ (heterogeneity).
# CONF: both harness cells: BEFORE point <= .25, AFTER point >= .75, NONE point >= .75 (validity).
# REJ: a valid cell with BEFORE Wilson low > .25 or AFTER Wilson high < .75. Else UNC.
import numpy as np, sys
rng=np.random.default_rng(1127); Z=1.96
def wil(k,n):
    p=k/n; d=1+Z*Z/n; c=(p+Z*Z/(2*n))/d; h=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d; return c-h,c+h
def arm(pm,B,kappa,N):
    tot=0
    for p in pm:  # per model
        if kappa is None: tot=tot+rng.binomial(3*B,p,N)
        else:
            q=rng.beta(max(p*kappa,1e-3),max((1-p)*kappa,1e-3),(N,3)); tot=tot+rng.binomial(B,q).sum(1)
    return tot
def sim(none,bef,aft,B,kappa=None,N=20000):
    n=2*3*B; conf=np.ones(N,bool); rej=np.zeros(N,bool)
    for h in range(2):
        kn=arm(none,B,kappa,N); kb=arm(bef,B,kappa,N); ka=arm(aft,B,kappa,N)
        v=kn/n>=.75; conf&=v&(kb/n<=.25)&(ka/n>=.75)
        lb,_=wil(kb,n); _,ua=wil(ka,n); rej|=v&((lb>.25)|(ua<.75))
    c=conf.mean(); r=(rej&~conf).mean(); return c,r,1-c-r
S={'law true 15/85 both':((.95,.95),(.15,.15),(.85,.85)),
   'law true 20/80 both':((.95,.95),(.20,.20),(.80,.80)),
   'hetero before 5/40':((.95,.95),(.05,.40),(.85,.85)),
   'hetero after 95/60':((.95,.95),(.15,.15),(.95,.60)),
   'presence only':((.95,.95),(.15,.15),(.15,.15)),
   'half order after 50':((.95,.95),(.15,.15),(.50,.50)),
   'no route effect':((.95,.95),(.95,.95),(.95,.95)),
   'after mild 70':((.95,.95),(.15,.15),(.70,.70))}
for B in (8,10,11):
  for kappa in (None,20):
    print(f'--- pooled n/arm/harness={6*B} kappa={kappa}')
    for k,v in S.items():
        c,r,u=sim(*v,B=B,kappa=kappa); print(f'{k:22s} CONF {c:.3f} REJ {r:.3f} UNC {u:.3f}')
