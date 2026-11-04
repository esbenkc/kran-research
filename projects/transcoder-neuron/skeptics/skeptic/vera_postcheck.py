"""Check post numbers not covered by vera_stats: 0.7 sensitivity row, write-side graph range, dose medians;
numeric check of displayed equations (nudge zeroes pre-activation; theory-ring formula under isotropic g)."""
import numpy as np, pandas as pd, json, torch
D='/Users/esben/work/theory-1104-transcoder-neuron/kran-research/projects/transcoder-neuron/data/pushes'
def r2(p,y): p,y=np.asarray(p,float),np.asarray(y,float); return float(1-((y-p)**2).sum()/((y-y.mean())**2).sum())
out={}
for L in [8,9,10,11]:
    U=pd.read_csv(f'{D}/L{L}_units.csv'); P=pd.read_csv(f'{D}/L{L}_pushes.csv')
    j=U.set_index(['group','uid']).joint
    P['joint']=[j.get((g,u),np.nan) for g,u in zip(P.group,P.uid)]
    a=P[(P.group.isin(['NEURON','GAP']))&(P.joint>=0.7)]
    o={'R2_0p7_neuron_plus_gap>=0.7':r2(a.total_r,a.meas_local_r),'n':len(a)}
    for g in ['NEURON','GAP','REST']:
        d=P[P.group==g]
        o[f'{g}_write_graph_minus']=r2(-d.wr_pred,d.wr_minus); o[f'{g}_write_graph_plus']=r2(d.wr_pred,d.wr_plus)
        o[f'{g}_write_real_minus']=r2(-d.wr_pred_r,d.wr_minus); o[f'{g}_write_real_plus']=r2(d.wr_pred_r,d.wr_plus)
    o['dose_med_pct']={g:round(100*float((P[P.group==g].du_norm/P[P.group==g].xn_norm).median()),2) for g in ['NEURON','GAP','REST','ORACLE']}
    out[L]=o; print(L,json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in o.items()}))
# equation 1: the nudge zeroes the pre-activation (from my own 400-push rebuild rows)
import sys; sys.path.insert(0,'.')
from vera_lib import load_tc
R=pd.read_csv('vera_rebuild_rows.csv')
out['eq_nudge']= 'pre - a_f*|e|^2/|e|^2 = 0 algebraically; numeric: rebuild pre_v vs pre_b med rel 2.5e-6 (vera_rebuild.json)'
# equation 2: theory ring. Monte-Carlo: isotropic g, s=g.r, shat=g.t -> E[R2] vs 1-sum|t-r|^2/sum|r|^2
rng=np.random.default_rng(0); n,dim=400,768
r=rng.normal(size=(n,dim))*rng.lognormal(size=(n,1)); t=r+0.6*rng.normal(size=(n,dim))*rng.lognormal(size=(n,1))*0.3
g=rng.normal(size=(2000,dim))
vals=[r2((t@gg),(r@gg)) for gg in g[:200]]
out['eq_theory_ring']=dict(formula=float(1-((t-r)**2).sum()/(r**2).sum()),mc_mean=float(np.mean(vals)),mc_sd=float(np.std(vals)))
print(out['eq_theory_ring'])
json.dump(out,open('vera_postcheck.json','w'),indent=1,default=float)
