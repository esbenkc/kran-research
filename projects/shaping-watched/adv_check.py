import numpy as np
rng=np.random.default_rng(0); G=8; maxdiff={'B':0,'O':0}; nflip=0; ntot=0
def std(x): return x.std(ddof=1)
for t in range(20000):
    flag=rng.random()<0.5; k=rng.integers(1,G)  # mixed groups only
    is_read=np.array([1]*k+[0]*(G-k),bool); rng.shuffle(is_read)
    Mr,Ms=rng.normal(12.6,1),rng.normal(-5.8,1)   # probe reads the action word (probe_M separation)
    def advof(r):
        return (r-r.mean())/(std(r)+1e-6)
    rP0=np.where(is_read, -1.0 if flag else 1.0, 0.0)
    rO=np.where(is_read, -1.0 if flag else 1.0, 0.6)
    r=np.where(is_read,1.0,0.6); pen=np.zeros(G)
    if flag:
        M=np.where(is_read,Mr,Ms); s=(M-M.mean())/(std(M)+1e-8)
        lam=0.4*min(1,std(r)/0.2); pen=lam*s
    rB=r-pen
    ntot+=1; nflip+= flag and (rB[is_read][0]<rB[~is_read][0])
    for n,x in (('B',rB),('O',rO)): maxdiff[n]=max(maxdiff[n],np.abs(advof(x)-advof(rP0)).max())
print('max |adv - adv_P0| over 20,000 mixed groups:',maxdiff)
print('probed groups where B puts read below solve: ', nflip)
