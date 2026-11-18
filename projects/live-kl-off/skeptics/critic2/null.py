import numpy as np
z=np.load('data/pool/test/cols.npz'); var=z['variant']; base=z['base_row']; x1=z['x1_orig']; pr=z['prompt_row']
n=np.where(var==0)[0]; n=n[np.argsort(base[n])]; x=np.where(var==1)[0]; x=x[np.argsort(base[x])]; k=~x1[n]; p=pr[n]
rng=np.random.default_rng(3)
for run,r in [('kl0_s2',600),('kl0_s1',800),('kl0_s0',1200)]:
    c=np.load(f'data/runs/{run}/reads/c_{r}.npy').astype(float); a=c[n]; b=c[x]; co=a.mean()
    obs=(a[k]-b[k]).mean()/co
    nul=[]
    for _ in range(2000):
        f=rng.integers(0,2,200)[p]*2-1  # prompt-clustered sign flip of the twin label
        nul.append(((a[k]-b[k])*f[k]).mean()/co)
    nul=np.array(nul); print(run,r,'obs %+.4f null sd %.4f max|null| %.4f p %.4f'%(obs,nul.std(),np.abs(nul).max(),(np.abs(nul)>=abs(obs)).mean()))
