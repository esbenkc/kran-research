# Pooled rule + per-model guard: CONF also requires no single model whose BEFORE Wilson low > .25
# or AFTER Wilson high < .75 (guard hit -> UNC, not REJ). Same scenarios as vera_power_pooled.py.
import numpy as np
rng=np.random.default_rng(1128); Z=1.96
def wil(k,n):
    p=k/n; d=1+Z*Z/n; c=(p+Z*Z/(2*n))/d; h=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d; return c-h,c+h
def one(p,B,kappa,N):
    if kappa is None: return rng.binomial(3*B,p,N)
    q=rng.beta(max(p*kappa,1e-3),max((1-p)*kappa,1e-3),(N,3)); return rng.binomial(B,q).sum(1)
def sim(none,bef,aft,B,kappa=None,N=20000):
    m=3*B; n=2*m; conf=np.ones(N,bool); rej=np.zeros(N,bool); guard=np.zeros(N,bool)
    for h in range(2):
        kn=sum(one(p,B,kappa,N) for p in none)
        kb=[one(p,B,kappa,N) for p in bef]; ka=[one(p,B,kappa,N) for p in aft]
        KB=sum(kb); KA=sum(ka)
        v=kn/n>=.75; conf&=v&(KB/n<=.25)&(KA/n>=.75)
        lb,_=wil(KB,n); _,ua=wil(KA,n); rej|=v&((lb>.25)|(ua<.75))
        for x in kb: guard|=wil(x,m)[0]>.25
        for x in ka: guard|=wil(x,m)[1]<.75
    conf&=~guard; r=(rej&~conf).mean(); c=conf.mean(); return c,r,1-c-r
S={'law true 15/85 both':((.95,.95),(.15,.15),(.85,.85)),
   'law true 20/80 both':((.95,.95),(.20,.20),(.80,.80)),
   'hetero before 5/40':((.95,.95),(.05,.40),(.85,.85)),
   'hetero before 5/50':((.95,.95),(.05,.50),(.85,.85)),
   'hetero after 95/60':((.95,.95),(.15,.15),(.95,.60)),
   'presence only':((.95,.95),(.15,.15),(.15,.15))}
for B in (8,10):
  for kappa in (None,20):
    print(f'--- guard, pooled n/arm/harness={6*B} (per model {3*B}) kappa={kappa}')
    for k,v in S.items():
        c,r,u=sim(*v,B=B,kappa=kappa); print(f'{k:22s} CONF {c:.3f} REJ {r:.3f} UNC {u:.3f}')
