"""Iris: how much does using population h_i (S = I) instead of the data-seed Hessian (S = X^T X/n, n = 1024)
move the law's closed-form regular term?  Gauss-Newton Hessian at w* is exact because the residual is 0.
Uses TRAIN shapes only (32,32,16,1) and (32,32,16,8)."""
import numpy as np, itertools, json
def hess(M,N,H,r,S,seed=0):
    g=np.random.default_rng(seed)
    U,_=np.linalg.qr(g.standard_normal((N,r))); V,_=np.linalg.qr(g.standard_normal((M,r)))
    A=np.zeros((H,M)); B=np.zeros((N,H)); A[:r]=V.T; B[:,:r]=U
    # d(BA) = dB A + B dA ; K = 1/2 tr(D S D^T); Hessian = J^T (S kron I_N) J with vec over (N x M)
    nB=N*H; nA=H*M; J=np.zeros((N*M,nB+nA))
    for k in range(nB):
        dB=np.zeros(N*H); dB[k]=1; J[:,k]=(dB.reshape(N,H)@A).ravel()
    for k in range(nA):
        dA=np.zeros(H*M); dA[k]=1; J[:,nB+k]=(B@dA.reshape(H,M)).ravel()
    W=np.kron(np.eye(N),S)  # row-major vec of N x M: index i*M+j ; D S D^T -> sum_i d_i S d_i^T
    Hm=J.T@W@J
    ev=np.linalg.eigvalsh(Hm); return ev[ev>1e-9]
def reg_term(h,nbeta,gamma,eps):
    g=gamma/(nbeta*h); c=eps*(nbeta*h+gamma); return float(np.sum(0.5/(1+g)/(1-c/4)))
out=[]
for (M,N,H,r) in [(32,32,16,1),(32,32,16,8)]:
    hp=hess(M,N,H,r,np.eye(M))
    X=np.random.default_rng(1).standard_normal((1024,M)); S=X.T@X/1024
    he=hess(M,N,H,r,S)
    for nbeta,q,c in itertools.product([100,1000,10000],[0.1,1,10,100],[0.01,0.1]):
        gamma=np.sqrt(nbeta*0.5/q); eps=c/(nbeta*hp.max()+gamma)
        rp=reg_term(hp,nbeta,gamma,eps); re=reg_term(he,nbeta,gamma,eps)
        from rrr_lambda import rrr_lambda as L
        lam=float(L(M,N,H,r))
        out.append(dict(shape=(M,N,H,r),nbeta=nbeta,q=q,c=c,diff_pts=100*(re-rp)/lam))
    print((M,N,H,r),'n_pos',len(hp),len(he),'h pop %.3f..%.3f emp %.3f..%.3f'%(hp.min(),hp.max(),he.min(),he.max()))
d=np.array([o['diff_pts'] for o in out]); print('regular-term shift (pts of lambda): min %.3f max %.3f median %.3f'%(d.min(),d.max(),np.median(d)))
json.dump(out,open('hess_sample_cov.json','w'),indent=0,default=str)
