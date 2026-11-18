# critic2 (Claude fallback) independent rebuild of s_P from raw per-row reads; own bootstrap seed 777.
import numpy as np, json, glob, re, sys
from pathlib import Path
D=Path(__file__).resolve().parents[2]/'data'
def pool(name):
    c=np.load(D/'pool_cols'/name/'cols.npz'); m=json.load(open(D/'pool_cols'/name/'meta.json'))
    var,base=c['variant'],c['base_row']; nb=m['n_base']
    idx={}
    for vi,v in enumerate(m['variants']):
        ii=np.where(var==vi)[0]; o=np.empty(nb,int); o[base[ii]]=ii; idx[v]=o
    n0=idx['none']; keep=~c['x1_orig'][n0].astype(bool); pr=c['prompt_row'][n0]
    u,inv=np.unique(pr,return_inverse=True)
    return idx,keep,inv,len(u)
rng=np.random.default_rng(777)
def sP(name,c,B=2000):
    idx,keep,inv,P=pool(name); cn=c[idx['none']].astype(float); cx=c[idx['xins']].astype(float)
    # per-prompt sums: n, sum cn, sum keep*(cn-cx), sum keep
    S=np.zeros((P,4)); np.add.at(S,inv,np.stack([np.ones_like(cn),cn,keep*(cn-cx),keep.astype(float)],1))
    f=lambda T:(T[...,2]/T[...,3])/(T[...,1]/T[...,0])
    W=rng.multinomial(P,np.full(P,1/P),size=B).astype(float)
    b=f(W@S); return f(S.sum(0)),np.percentile(b,2.5),np.percentile(b,97.5)
def run(rd,name='test'):
    fs=sorted([f for f in glob.glob(str(rd/"c_*.npy")) if re.fullmatch(r"c_\d+\.npy",Path(f).name)],key=lambda f:int(re.search(r"c_(\d+)\.npy",f).group(1)))
    for f in fs:
        s=int(re.search(r'c_(\d+)',f).group(1)); v,lo,hi=sP(name,np.load(f)); print(f'  {s:5d} s_P {v:+.4f} [{lo:+.4f},{hi:+.4f}]')
for r in ['runs/kl0_s0/reads','runs/posctl_d086_s0/reads','runs/replay_s0/reads','parent/live_k4/reads','parent/cachedkl_s0/reads','parent/cached_d010_s0/reads']:
    print(r); run(D/r)
print('placebo let'); 
for f in ['c_0.npy','c_kl0_s0_1200.npy']: v,lo,hi=sP('test_let',np.load(D/'reads/let'/f)); print(' ',f,f'{v:+.4f} [{lo:+.4f},{hi:+.4f}]')
