import numpy as np
from math import comb
cnt={'h':(18,18),'g1':(20,20),'g5':(20,20),'mini':(13,15),'nano':(16,8)}  # (uF, lR) of 20
def q975(c):
    dist=np.array([1.0]);off=0
    for f,r in c.values():
        pf=np.array([comb(20,k)*(f/20)**k*(1-f/20)**(20-k) for k in range(21)])
        pr=np.array([comb(20,k)*(r/20)**k*(1-r/20)**(20-k) for k in range(21)])
        dist=np.convolve(dist,np.convolve(pf,pr[::-1]));off+=20
    v=(np.arange(len(dist))-off)/100; c=np.cumsum(dist)
    return v[np.searchsorted(c,0.975-1e-12)], v[np.searchsorted(c,0.025-1e-12)]
print('base',q975(cnt))
for m in cnt:
    f,r=cnt[m]
    if r>0: c=dict(cnt); c[m]=(f,r-1); print('lR-1',m,q975(c))
    if f<20: c=dict(cnt); c[m]=(f+1,r); print('uF+1',m,q975(c))
