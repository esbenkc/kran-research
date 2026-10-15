import numpy as np
rng=np.random.default_rng(1)
# generic-input over-param check: need N >= D distinct generic inputs so data does not cap rank
p=3;d=6;D=d*(d+1)//2;N=200;X=rng.standard_normal((d,N));P=np.eye(N)-1/N
def J(K,vonly=False,X=X):
    N=X.shape[1];P=np.eye(N)-1/N
    W=rng.standard_normal((d,K));V=rng.standard_normal((p,K));S=W.T@X
    JV=np.zeros((p,N,p,K));JW=np.zeros((p,N,d,K))
    for k in range(p):
        JV[k,:,k,:]=(S**2).T; JW[k]=np.einsum('in,jn->nij',X,2*V[k][:,None]*S)
    M=JV.reshape(p,N,-1) if vonly else np.concatenate([JV.reshape(p,N,-1),JW.reshape(p,N,-1)],axis=2)
    return np.linalg.matrix_rank(np.einsum('mn,knq->kmq',P,M).reshape(p*N,-1))
for K in [2,4,8,20,40]:
    print(f"generic p=3 d=6 K={K}: rank={J(K)}  min(K(d+p-1), p*D)={min(K*(d+p-1),p*D)}  (note centring removes const: p*(D-1)={p*(D-1)})")
# lazy: V-only vs full on one-hot data, p=7 train frac 0.5
p=7;d=14;pairs=[(a,b) for a in range(p) for b in range(p)];idx=rng.choice(49,25,replace=False)
X=np.zeros((d,25))
for n,i in enumerate(idx): a,b=pairs[i];X[a,n]=1;X[p+b,n]=1
for K in [5,10,30,60]:
    print(f"lazy p=7 N=25 K={K}: Vonly rank={J(K,True,X)} p*min(N-1,K)={p*min(24,K)} | full rank={J(K,False,X)}")
