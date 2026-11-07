# Vera: add Felix/Kasper's title-literal clause: clean flag (key-logged) - clean flag (no notice), agent-equal, 90% CI lo > 0.
import numpy as np
exec(open('vera_power_tove.py').read().split("print('d each way")[0])
def run2(d,n,R=20000):
    N=P0; S=np.array([shift(p,d,'flag') for p in P0]); M=np.array([shift(p,d,'key') for p in P0])
    dr=lambda P: np.stack([rng.multinomial(n,P[i],R) for i in range(5)],1)/n
    cn,cs,cm=dr(N),dr(S),dr(M)
    def ci(a,b):
        C=(a-b).mean(1); se=np.sqrt((a*(1-a)/n+b*(1-b)/n).sum(1))/5; return C-1.645*se
    cross=(ci(cs[...,1],cm[...,1])>0)&((cs[...,0]+cs[...,2]).mean(1)<(cm[...,0]+cm[...,2]).mean(1))
    lit=ci(cs[...,1],cn[...,1])>0
    return cross.mean(), lit.mean(), (cross&lit).mean()
print('d | n | P(crossover clause) P(literal clause) P(both)')
for n in (30,40):
    for d in (0,.05,.10,.15,.20): print(d,n,*[round(x,3) for x in run2(d,n)])
