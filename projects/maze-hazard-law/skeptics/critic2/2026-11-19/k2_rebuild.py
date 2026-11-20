import json,glob,math,collections,sys
import numpy as np
from scipy import stats, optimize
def runs(agent):
    out=[]
    for f in sorted(glob.glob(f'data/{agent}/raw/maze/*.json')):
        d=json.load(open(f))
        seen=[];ent=False
        for a in d['actions']:
            p=a.get('path') or ''
            if a['tool']=='fetch' and p.startswith('/m/') and a.get('status')==200:
                ent=True
                if p not in seen: seen.append(p)
        out.append((f.split('/')[-1],ent,len(seen),d['end']))
    return out
for agent in ['qwen3-8b-nothink','gpt-oss-20b-low']:
    R=runs(agent); E=[r for r in R if r[1]]
    print(agent,'runs',len(R),'entered',len(E), collections.Counter(r[3] for r in E))
    G=collections.Counter(); N=collections.Counter()
    Ds=[];ev=[];Dc=[]
    for _,_,D,end in E:
        gave = end in ('answer','leave_get')
        for d in range(1,(D if end!='cap' else D-1)+1): N[d]+=1
        if gave: G[D]+=1
        Ds.append(D); ev.append(gave); Dc.append(D if end!='cap' else D-1)
    for d in (1,2,3): print(f' page {d}: {G[d]}/{N[d]}')
    bins=[(1,3),(4,10),(11,30),(31,999)]
    tab=[(sum(G[d] for d in range(a,b+1)),sum(N[d] for d in range(a,b+1))) for a,b in bins]
    print(' bins',tab)
    g=sum(G.values()); e=sum(N.values()); h=g/e
    print(f' G {g} E {e} h_all {h:.4f}  G3 {tab[0][0]} E3 {tab[0][1]} h3 {tab[0][0]/tab[0][1]:.4f}')
    # KM median
    S=1.0; med=None
    for d in range(1,max(Ds)+1):
        at=sum(1 for x in Ds if x>=d); ev_d=sum(1 for x,v in zip(Ds,ev) if x==d and v)
        if at: S*=1-ev_d/at
        if S<=0.5 and med is None: med=d
    print(' KM median',med,' D list',sorted(Ds))
    # Pearson chi2
    tb=[t for t in tab if t[1]>0]
    chi=sum((gg-n*h)**2/(n*h)+((n-gg)-n*(1-h))**2/(n*(1-h)) for gg,n in tb)
    print(f' Pearson {chi:.2f} df {len(tb)-1} p {stats.chi2.sf(chi,len(tb)-1):.3g}' if g else ' no events')
    if g:
        def nll(th,ds=None,evs=None):
            ds=ds or Dc; evs=evs or ev
            q=1/(1+math.exp(-th[0])); k=math.exp(th[1]); ll=0
            for D,v in zip(ds,evs):
                pass
                for d in range(1,D+1):
                    hz=1-q**(d**k-(d-1)**k); hz=min(max(hz,1e-12),1-1e-12)
                    ll+= math.log(hz) if (d==D and v) else math.log(1-hz)
            return -ll
        best=min((optimize.minimize(nll,[a,b],method='Nelder-Mead',options={'xatol':1e-8,'fatol':1e-10,'maxiter':4000}) for a in (-2,0,2) for b in (-1,0,1)),key=lambda r:r.fun)
        llg=-nll([math.log((1-h)/h),0.0])
        LR=2*(-best.fun-llg)
        print(f' Weibull k {math.exp(best.x[1]):.3f} LR {LR:.2f} p {stats.chi2.sf(LR,1):.3g}')
        # leave out the capped run
        keep=[(D,v) for (n,_,D,end),v in zip(E,ev) if end!='cap']
        if len(keep)<len(Ds):
            ds,evs=zip(*keep); gg=sum(evs); ee=sum(ds); hh=gg/ee
            b2=min((optimize.minimize(lambda t:nll(t,ds,evs),[a,b],method='Nelder-Mead',options={'xatol':1e-8,'fatol':1e-10,'maxiter':4000}) for a in (-2,0,2) for b in (-1,0,1)),key=lambda r:r.fun)
            LR2=2*(-b2.fun+nll([math.log((1-hh)/hh),0.0],ds,evs))
            print(f' w/o cap run: k {math.exp(b2.x[1]):.3f} p {stats.chi2.sf(LR2,1):.3g}')
    # page1 vs page2 Fisher
    if agent.startswith('qwen'):
        print(' fisher p1 vs p2',stats.fisher_exact([[G[1],N[1]-G[1]],[G[2],N[2]-G[2]]])[1])
