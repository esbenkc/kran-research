"""Vera extras on the end read: start-adapter paired change, bootstrap-seed stability, per-draw heterogeneity,
probe-on cheating clause, half-split stability. Own code, raw npz only."""
import numpy as np, json
from vera_core import prompt_groups, boot_weights, ci
E='/Users/esben/work/oracle-1112-matched-trace/kran-research/projects/matched-trace-powered/data/reads/raw/end_tf.npz'
S='/Users/esben/work/oracle-1111-matched-trace/kran-research/projects/matched-trace/data/reads/raw/start_test.npz'
M='/Users/esben/work/oracle-1111-matched-trace/kran-research/projects/matched-trace/data/reads/raw/joint_s0_final.npz'
out={}
def per_prompt(z, lnb_key):
    c=z['c_nb'].astype(float); cl=z[lnb_key].astype(float) if lnb_key in z else c
    w=np.exp(z['lp_lb_nb'].astype(float).sum(1)-z['lp_nb'].astype(float).sum(1))
    p=z['pX_off'].astype(float); DP=np.minimum(p+.1,.99)-p
    u,inv=prompt_groups(z['prompt_idx']); G=len(u)
    def s(x):
        a=np.zeros(G); np.add.at(a,inv,x); return a
    return u, dict(n=s(np.ones_like(c)), c=s(c), w=s(w), wcl=s(w*cl), tv=s(np.maximum(0,1-w)), yp=s(DP*(c-z['c_X'])), dp=s(DP))
def shares(g):
    co=g['c']/g['n']; yR=co-g['wcl']/g['w']; D=g['tv']/g['n']
    return yR/(D*co), (g['yp']/g['n'])/((g['dp']/g['n'])*co), yR, g['yp']/g['n']
ze=np.load(E); zs=np.load(S); zm=np.load(M)
ue,ge=per_prompt(ze,'c_lb_nb'); us,gs=per_prompt(zs,'none'); um,gm=per_prompt(zm,'none')
assert (ue==us).all() and (ue==um).all(), 'prompt sets differ'
G=len(ue); W=boot_weights(G,2000,5)
def bs(g,Wv): return shares({k:Wv@v for k,v in g.items()})
one=np.ones(G)
for name,g in [('end',ge),('start_textonly',gs),('674_textonly',gm)]:
    sR,sP,yR,yP=bs(g,one); bR,bP,byR,byP=bs(g,W)
    out[name]=dict(s_R=[sR]+ci(bR), s_P=[sP]+ci(bP), y_P_pts=[100*yP]+[100*x for x in ci(byP)])
for ref,gr in [('start',gs),('674',gm)]:
    eR,eP,_,eyP=bs(ge,W); rR,rP,_,ryP=bs(gr,W); pR,pP,_,pyP=bs(ge,one); qR,qP,_,qyP=bs(gr,one)
    out['end_minus_'+ref]=dict(d_s_P=[pP-qP]+ci(eP-rP), d_s_R=[pR-qR]+ci(eR-rR), d_y_P_pts=[100*(pyP-qyP)]+[100*x for x in ci(eyP-ryP)])
# bootstrap seed stability of the status numbers
st=[]
for sd in range(6):
    Wv=boot_weights(G,2000,100+sd); bR,bP,_,_=bs(ge,Wv); st.append(dict(seed=sd,sR_hi=ci(bR)[1],sP_lo=ci(bP)[0],gap_lo=ci(bP-bR)[0]))
out['boot_seed_stability']=st
# per-draw heterogeneity of y_R (LR, self-normalised per draw)
c=ze['c_nb'].astype(float); cl=ze['c_lb_nb'].astype(float); w=np.exp(ze['lp_lb_nb'].astype(float).sum(1)-ze['lp_nb'].astype(float).sum(1)); sd_=ze['seed']
ys=[];vs=[]
for s in np.unique(sd_):
    m=sd_==s; y=c[m].mean()-(w[m]*cl[m]).sum()/w[m].sum()
    r=(c[m]-c[m].mean())-(w[m]*(cl[m]-(w[m]*cl[m]).sum()/w[m].sum()))/w[m].mean()
    ys.append(y); vs.append(r.var()/m.sum())
ys=np.array(ys); vs=np.array(vs); mu=(ys/vs).sum()/(1/vs).sum()
out['per_draw_yR']=dict(n=len(ys), chi2=float(((ys-mu)**2/vs).sum()), dof=len(ys)-1, min_pts=100*ys.min(), max_pts=100*ys.max())
# halves by draw (0-15 vs 16-31)
for nm,m in [('draws_0_15',sd_<16),('draws_16_31',sd_>=16)]:
    y=c[m].mean()-(w[m]*cl[m]).sum()/w[m].sum(); D=np.maximum(0,1-w[m]).mean()
    out[nm+'_s_R']=y/(D*c[m].mean())
# probe-on cheating clause: E_leaky[c_leaky] on end (LR) and plain c_lb, vs start c_lb
out['probe_on_cheat_end_LR']=float((w*cl).sum()/w.sum()); out['probe_on_cheat_end_plain']=float(ze['c_lb'].mean())
out['probe_on_cheat_start_plain']=float(zs['c_lb'].mean()); out['probe_on_cheat_674_plain']=float(zm['c_lb'].mean())
out['probe_on_move_pts_vs_start']=100*(out['probe_on_cheat_end_plain']-out['probe_on_cheat_start_plain'])
json.dump(out,open('vera_extra.json','w'),indent=1,default=float)
for k,v in out.items(): print(k,v)
# start-adapter paired gap (innate planted switch vs trace at the pilot adapter, text-only trace form)
Wv=boot_weights(G,2000,9); bR,bP,_,_=bs(gs,Wv); sR,sP,_,_=bs(gs,one)
out2=dict(start_gap=[sP-sR]+ci(bP-bR))
eR,eP,_,_=bs(ge,Wv); rR,rP,_,_=bs(gs,Wv)
out2['end_gap_minus_start_gap']=[(bs(ge,one)[1]-bs(ge,one)[0])-(sP-sR)]+ci((eP-eR)-(rP-rR))
print(out2); json.dump(out2,open('vera_extra2.json','w'),indent=1,default=float)
