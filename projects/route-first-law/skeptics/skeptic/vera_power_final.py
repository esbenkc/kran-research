# Final rule (Felix v1.2 #31185): per harness pooled over 2 models, n=30/model/arm (60 pooled).
# CONF: both harnesses: pooled BEFORE <= .25, AFTER >= .75, NONE >= .75 (validity; exposure assumed met),
#       and per-model guard: each model BEFORE <= .40 and AFTER >= .60 (point). Guard miss -> UNC.
# REJ: a valid harness with pooled BEFORE Wilson low > .25 or AFTER Wilson high < .75 (and not CONF).
import numpy as np
rng=np.random.default_rng(1129); Z=1.96
def wil(k,n):
    p=k/n; d=1+Z*Z/n; c=(p+Z*Z/(2*n))/d; h=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d; return c-h,c+h
def one(p,B,kappa,N):
    if kappa is None: return rng.binomial(3*B,p,N)
    q=rng.beta(max(p*kappa,1e-3),max((1-p)*kappa,1e-3),(N,3)); return rng.binomial(B,q).sum(1)
def sim(none,bef,aft,B=10,kappa=None,N=20000):
    m=3*B; n=2*m; conf=np.ones(N,bool); rej=np.zeros(N,bool)
    for h in range(2):
        kn=sum(one(p,B,kappa,N) for p in none)
        kb=[one(p,B,kappa,N) for p in bef]; ka=[one(p,B,kappa,N) for p in aft]
        KB=sum(kb); KA=sum(ka); v=kn/n>=.75
        conf&=v&(KB/n<=.25)&(KA/n>=.75)
        for x in kb: conf&=x/m<=.40
        for x in ka: conf&=x/m>=.60
        lb,_=wil(KB,n); _,ua=wil(KA,n); rej|=v&((lb>.25)|(ua<.75))
    c=conf.mean(); r=(rej&~conf).mean(); return c,r,1-c-r
S={'law true 15/85':((.95,.95),(.15,.15),(.85,.85)),
   'law true 20/80':((.95,.95),(.20,.20),(.80,.80)),
   'law strong 10/90':((.95,.95),(.10,.10),(.90,.90)),
   'hetero before 5/40':((.95,.95),(.05,.40),(.85,.85)),
   'hetero before 5/45':((.95,.95),(.05,.45),(.85,.85)),
   'hetero after 95/60':((.95,.95),(.15,.15),(.95,.60)),
   'hetero after 95/55':((.95,.95),(.15,.15),(.95,.55)),
   'presence only 15/15':((.95,.95),(.15,.15),(.15,.15)),
   'half order after 50':((.95,.95),(.15,.15),(.50,.50)),
   'after mild 70':((.95,.95),(.15,.15),(.70,.70)),
   'no route effect':((.95,.95),(.95,.95),(.95,.95)),
   'NONE invalid 70':((.70,.70),(.15,.15),(.85,.85))}
for kappa in (None,20):
    print(f'--- FINAL rule, n=30/model/arm, 60 pooled, kappa={kappa}')
    for k,v in S.items():
        c,r,u=sim(*v,kappa=kappa); print(f'{k:22s} CONF {c:.3f} REJ {r:.3f} UNC {u:.3f}')
