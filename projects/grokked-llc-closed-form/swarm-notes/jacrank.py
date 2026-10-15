# Critic check: rank of Jacobian of theta -> centred outputs on modular-addition data.
# 2*lambda = this rank at a generic smooth true solution (paper Thm 4.1 logic, restricted to the data).
import numpy as np
rng=np.random.default_rng(0)
def jac_rank(p,K,frac=1.0,generic_inputs=False):
    d=2*p
    pairs=[(a,b) for a in range(p) for b in range(p)]
    if frac<1: pairs=[pairs[i] for i in rng.choice(len(pairs),int(frac*len(pairs)),replace=False)]
    N=len(pairs)
    if generic_inputs: X=rng.standard_normal((d,N))
    else:
        X=np.zeros((d,N))
        for n,(a,b) in enumerate(pairs): X[a,n]=1;X[p+b,n]=1
    W=rng.standard_normal((d,K));V=rng.standard_normal((p,K))
    S=W.T@X                      # K x N preacts
    P=np.eye(N)-1/N              # centring
    # outputs O[k,n]=sum_j V[k,j] S[j,n]^2 ; Jacobian rows (k,n)
    JV=np.zeros((p,N,p,K)); JW=np.zeros((p,N,d,K))
    for k in range(p):
        JV[k,:,k,:]=(S**2).T
        JW[k]=np.einsum('in,jn->nij',X,2*V[k][:,None]*S)
    J=np.concatenate([JV.reshape(p,N,-1),JW.reshape(p,N,-1)],axis=2)
    J=np.einsum('mn,knq->kmq',P,J).reshape(p*N,-1)
    return np.linalg.matrix_rank(J),N
for p in [7,11]:
    for K in [2,4,8,16]:
        r,N=jac_rank(p,K); rg,_=jac_rank(p,K,generic_inputs=True)
        print(f"p={p} K={K} N={N} rank_onehot={r} rank_generic={rg} K(3p-1)={K*(3*p-1)} K(3p-2)={K*(3*p-2)} cap p(N-1)={p*(N-1)}")
p=7; d=14
for K in [40,80,200]:
    r,N=jac_rank(p,K); print(f"overparam p=7 K={K}: rank_onehot={r}, pd(d+1)/2={p*d*(d+1)//2}, p(N-1)={p*(N-1)}")
