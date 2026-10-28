import json, numpy as np, sys
sys.argv=['x','projects/decoy-capability/raw']
src=open('projects/decoy-capability/skeptics/critic2/c2_rebuild.py').read().split("rows = {}")[0]; exec(src)
D=json.load(open('projects/decoy-capability/skeptics/critic2/c2_rebuild.json')); fit=D['fit']
# cluster: strong five
strong=sorted(fit,key=lambda m:D['rows'][m]['x'])[3:]
X=[D['rows'][m]['x'] for m in strong]; rk=max(X)-min(X)
d=[D['rows'][m]['static']['u']-D['rows'][m]['stateful']['u'] for m in strong]
b,se=ols(X,d); t3=2.3533634348018264
print('strong5 I %+.3f [%+.3f,%+.3f] range %.3f'%(b*rk,(b-t3*se)*rk,(b+t3*se)*rk,rk))
# two-level bootstrap
data={}
for m in fit:
    nb=load(m,'nobait'); st=load(m,'static'); sf=load(m,'stateful')
    xs=np.array([sum(r['subgoals'])/len(r['subgoals']) for r in nb.values()])
    seeds=sorted(set(st)&set(sf))
    ts=np.array([score(st[s])[0] for s in seeds]); us_=np.array([score(st[s])[1] for s in seeds])
    tf=np.array([score(sf[s])[0] for s in seeds]); uf_=np.array([score(sf[s])[1] for s in seeds])
    data[m]=(xs,ts,us_,tf,uf_)
rng=D['static_change'][0]; R0=max(D['rows'][m]['x'] for m in fit)-min(D['rows'][m]['x'] for m in fit)
g=np.random.default_rng(0); S=[];I=[];skip=0
for _ in range(10000):
    ms=g.choice(len(fit),len(fit)); xx=[];a=[];c=[]; ok=True
    for k in ms:
        xs,ts,us_,tf,uf_=data[fit[k]]
        xx.append(xs[g.integers(0,len(xs),len(xs))].mean())
        idx=g.integers(0,len(ts),len(ts))
        if ts[idx].sum()==0 or tf[idx].sum()==0: ok=False;break
        a.append(us_[idx].sum()/ts[idx].sum()); c.append(uf_[idx].sum()/tf[idx].sum())
    if not ok or np.ptp(xx)==0: skip+=1; continue
    xx=np.array(xx); bs=np.polyfit(xx,a,1)[0]; bf=np.polyfit(xx,c,1)[0]
    S.append(bs*R0); I.append((bs-bf)*R0)
S=np.array(S);I=np.array(I)
sl,sh=np.percentile(S,[2.5,97.5]); il,ih=np.percentile(I,[2.5,97.5])
print('boot static [%+.3f,%+.3f] I [%+.3f,%+.3f] skipped %d'%(sl,sh,il,ih,skip))
pt=D['static_change'][0]
conf= ih<0 and sh<0 and pt<=-0.10; rej= sl>-0.10 or il>-0.10
print('orig-rule status', 'REJECTED' if rej else ('CONFIRMED' if conf else 'UNCONFIRMED'))
