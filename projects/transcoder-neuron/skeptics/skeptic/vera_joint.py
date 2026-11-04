"""Joint label-shuffle test over 4 layers (own code): statistic = sum of per-layer gaps standardised by each layer's null sd.
Also counts of 'big misses' |pred-meas| > 1 logit per group."""
import numpy as np, pandas as pd, json
D='/Users/esben/work/theory-1104-transcoder-neuron/kran-research/projects/transcoder-neuron/data/pushes'
rng=np.random.default_rng(2024)
def r2(p,y): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
lay={}
for L in [8,9,10,11]:
    P=pd.read_csv(f'{D}/L{L}_pushes.csv'); d=P[P.group.isin(['NEURON','REST'])]
    uids=d.uid.values; uq,inv=np.unique(uids,return_inverse=True)
    lab=d.groupby('uid').group.first().reindex(uq).values=='NEURON'
    p,y=d.total_r.values,d.meas_local_r.values
    def gap(lb, p=p, y=y, inv=inv):
        m=lb[inv]; return r2(p[m],y[m])-r2(p[~m],y[~m])
    lay[L]=(gap,lab)
    print(L,'big misses >1 logit', {g:f"{int((np.abs(P[P.group==g].total_r-P[P.group==g].meas_local_r)>1).sum())}/{int((P.group==g).sum())}" for g in ['NEURON','GAP','REST','ORACLE']})
B=2000
obs={L:lay[L][0](lay[L][1]) for L in lay}
nulls={L:np.array([lay[L][0](rng.permutation(lay[L][1])) for _ in range(B)]) for L in lay}
sd={L:nulls[L].std() for L in lay}
S_obs=sum(obs[L]/sd[L] for L in lay); S_null=sum(nulls[L]/sd[L] for L in lay)
pos_null=sum((nulls[L]>0).astype(int) for L in lay)
res=dict(obs_gap=obs,per_layer_p_two={L:float((np.abs(nulls[L])>=abs(obs[L])).mean()) for L in lay},
 joint_p_one=float(((S_null>=S_obs).sum()+1)/(B+1)), all4_positive_null_rate=float((pos_null==4).mean()),
 min_gap_null_p=float(((np.min(np.stack([nulls[L]/sd[L] for L in lay]),0)>=min(obs[L]/sd[L] for L in lay)).sum()+1)/(B+1)))
print(json.dumps(res,default=float))
json.dump(res,open('vera_joint.json','w'),default=float,indent=1)
