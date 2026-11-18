# critic2 (Claude fallback) independent rebuild of s_P and onset rule from raw c_*.npy
import numpy as np, json, sys, glob, os
P='data/pool/test/cols.npz'
z=np.load(P); var=z['variant']; base=z['base_row']; pr=z['prompt_row']; x1=z['x1_orig']; lw=z['logw']
none=var==0; xi=var==1
# pair rows by base_row
bn=base[none]; bx=base[xi]
on=np.argsort(bn); ox=np.argsort(bx)
assert (bn[on]==bx[ox]).all()
idx_n=np.where(none)[0][on]; idx_x=np.where(xi)[0][ox]
keep=~x1[idx_n]
prompt=pr[idx_n]
print('rows',len(idx_n),'x1 rows',int((~keep).sum()),'prompts',len(np.unique(prompt)),'logw range',lw.min(),lw.max())
def sP(c, sel=None):
    cn=c[idx_n]; cx=c[idx_x]
    return (cn[keep]-cx[keep]).mean()/cn.mean()
rng=np.random.default_rng(12345)
up=np.unique(prompt); groups=[np.where(prompt==p)[0] for p in up]
def boot(c, B=2000):
    cn=c[idx_n]; cx=c[idx_x]; d=np.where(keep,cn-cx,0.0); k=keep.astype(float)
    S_d=np.array([d[g].sum() for g in groups]); S_k=np.array([k[g].sum() for g in groups]); S_c=np.array([cn[g].sum() for g in groups]); n_g=np.array([len(g) for g in groups])
    out=np.empty(B)
    for b in range(B):
        w=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
        out[b]=(w@S_d/(w@S_k))/(w@S_c/(w@n_g))
    return out
res={}
for run in ['kl0_s0','kl0_s1','kl0_s2']:
    d=f'data/runs/{run}/reads'; files=sorted(glob.glob(d+'/c_*.npy'),key=lambda f:int(f.split('_')[-1][:-4]))
    stored={j['step']:j for j in map(json.loads,open(f'data/runs/{run}/reads.jsonl'))}
    rows=[]
    for f in files:
        r=int(f.split('_')[-1][:-4]); c=np.load(f).astype(np.float64)
        s=sP(c); bs=boot(c); lo,hi=np.percentile(bs,[2.5,97.5])
        rows.append((r,s,lo,hi,bs,c))
        print(run,r,f'{s:+.4f} [{lo:+.4f},{hi:+.4f}] stored {stored[r]["s_P"]:+.4f} diff {abs(s-stored[r]["s_P"]):.1e}')
    # onset rule
    r0=rows[0]; c0=r0[5]
    def paired_lo(c):
        cn0=c0[idx_n];cx0=c0[idx_x];cn=c[idx_n];cx=c[idx_x]
        out=[]
        for b in range(2000):
            sel=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
            kk=keep[sel]
            a=(cn[sel][kk]-cx[sel][kk]).mean()/cn[sel].mean(); a0=(cn0[sel][kk]-cx0[sel][kk]).mean()/cn0[sel].mean()
            out.append(a-a0)
        return np.percentile(out,2.5)
    q=[]
    for (r,s,lo,hi,bs,c) in rows[1:]:
        q.append(s>=0.30 and s>r0[3] and paired_lo(c)>0 if s>=0.30 else False)
    onset=None
    for i in range(len(q)-1):
        if q[i] and q[i+1]: onset=rows[i+1][0]; break
    last=rows[-1]
    print(run,'qualifying',[rows[i+1][0] for i,v in enumerate(q) if v],'onset',None if onset is None else f'({onset-100},{onset}]','last read',last[0])
    res[run]=dict(onset=onset,last=last[0],sP={r:s for r,s,*_ in rows})
common=min(v['last'] for v in res.values())
vals=np.array([res[k]['sP'][common] for k in res])
from math import sqrt
t=4.302652729911275 # t_{0.975,2}
m=vals.mean(); se=vals.std(ddof=1)/sqrt(3)
print('common read',common,'vals',vals.round(4),f'seed-level {m:.3f} [{m-t*se:.3f},{m+t*se:.3f}]')
