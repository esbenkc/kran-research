# Vera: power of the frozen prediction rule at warning-override's unwarned flag-present bases.
import numpy as np
rng=np.random.default_rng(0)
M=['haiku','gem31','gem35','mini','nano']
fb=np.array([.70,.65,.975,.0,.175]); kb=np.array([.425,.875,.475,.975,.675])
def sim(dflag,dkey,n=40,R=20000,mode='add'):
    if mode=='add':
        fs=np.clip(fb+dflag,0,1); ks=np.clip(kb-dkey,0,1)
    else: # fraction of remaining headroom
        fs=fb+dflag*(1-fb); ks=kb*(1-dkey)
    F0=rng.binomial(n,fb,(R,5))/n; F1=rng.binomial(n,fs,(R,5))/n
    K0=rng.binomial(n,kb,(R,5))/n; K1=rng.binomial(n,ks,(R,5))/n
    pf=(F1-F0).mean(1); pk=(K0-K1).mean(1); per=((F1-F0)>=0.10).sum(1)
    return ((pf>=.15)&(pk>=.15)).mean(), (per>=3).mean(), ((pf>=.15)&(pk>=.15)&(per>=3)).mean()
print('true add-shift  P(pooled both)  P(3of5)  P(scoped clauses)')
for d in [0,.1,.15,.2,.25,.3,.4]:
    print('add',d,*[round(x,3) for x in sim(d,d)])
for d in [.2,.3,.5,.7]:
    print('headroom-frac',d,*[round(x,3) for x in sim(d,d,mode='frac')])
