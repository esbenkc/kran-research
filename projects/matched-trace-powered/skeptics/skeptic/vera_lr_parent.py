"""Check the likelihood-ratio (LR) trace read on the PARENT raw npz (11-11), my own code."""
import numpy as np, json, sys
from vera_core import *
P='/Users/esben/work/oracle-1111-matched-trace/kran-research/projects/matched-trace/data/reads/raw/'
for name in ['joint_s0_final','start_test']:
    z=np.load(P+name+'.npz')
    w=np.exp(z['lp_lb_nb'].sum(1)-z['lp_nb'].sum(1))
    c=z['c_nb'].astype(float)
    u,inv=prompt_groups(z['prompt_idx']); G=len(u)
    plain=c-z['c_lb']; plain_text=c-z['c_nlb']
    lr_raw=c*(1-w)
    b=c.mean(); lr=(c-b)*(1-w)
    D=np.maximum(0,1-w)
    print(name,'rows',len(c),'E[w]',w.mean(),'w range',w.min(),w.max(),'P(w>2)',(w>2).mean())
    for nm,a in [('plain',plain),('plain_text',plain_text),('lr_raw',lr_raw),('lr_b',lr)]:
        print(f'  {nm:10s} mean {a.mean()*100:+.3f} pts  rowSD {a.std():.4f}')
    # clustered bootstrap of s_R with LR
    Wt=boot_weights(G,2000,1)
    (sy,sd,sc,sp),cnt=cluster_sums(inv,G,lr,D,c,plain)
    n=Wt@cnt
    yb=(Wt@sy)/n; Db=(Wt@sd)/n; cb=(Wt@sc)/n; pb=(Wt@sp)/n
    s=yb/(Db*cb); sp_=pb/(Db*cb)
    s0=lr.mean()/(D.mean()*c.mean())
    print('  s_R LR',round(s0,4),'CI',np.round(ci(s),4),'HW',round((ci(s)[1]-ci(s)[0])/2,4),'| s_R plain',round(plain.mean()/(D.mean()*c.mean()),4),'CI',np.round(ci(sp_),4))
    # w-check: TV both directions
    print('  D_R',D.mean(), ' E[w]-1 CI', np.round(ci((Wt@cluster_sums(inv,G,w)[0][0])/n -1),4))
