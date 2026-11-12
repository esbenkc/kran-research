import numpy as np, json
rng=np.random.default_rng(0); d=1024; r=16
out=[]
for t in range(200):
    c=rng.standard_normal(d); c/=np.linalg.norm(c)
    Ms=[]
    for dout in (2048,1024,1024):
        A=rng.uniform(-1/32,1/32,(r,d))  # kaiming-uniform bound 1/sqrt(d)
        g=rng.standard_normal(dout)
        B=np.outer(g,A@c)   # B-only, cue-driven gradient, rank-1
        Ms.append(B@A)
    M=np.vstack(Ms)
    v1=np.linalg.svd(M,full_matrices=False)[2][0]
    e_true=(v1@c)**2
    U=rng.standard_normal((1000,d)); U/=np.linalg.norm(U,axis=1,keepdims=True)
    e_null=(U@v1)**2
    out.append((e_true,(e_null>=e_true).sum()+1))
e=np.array([o[0] for o in out]); rk=np.array([o[1] for o in out])
print(json.dumps({"e1_mean":e.mean(),"e1_p5":np.percentile(e,5),"e1_p95":np.percentile(e,95),"r_over_d":r/d,"P_rank1":(rk==1).mean(),"P_rank_le10":(rk<=10).mean(),"null_p99":float(np.percentile(np.random.default_rng(1).beta(.5,(d-1)/2,100000),99))},indent=0))
