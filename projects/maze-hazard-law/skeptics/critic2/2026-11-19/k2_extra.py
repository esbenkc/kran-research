import json,glob,collections,math
from scipy import stats
import numpy as np
def rec(agent,arm):
    out=[]
    for f in sorted(glob.glob(f'data/{agent}/raw/{arm}/*.json')):
        d=json.load(open(f)); seen=[]
        for a in d['actions']:
            p=a.get('path') or ''
            if a['tool']=='fetch' and p.startswith('/m/') and a.get('status')==200 and p not in seen: seen.append(p)
        out.append(dict(f=f.split('/')[-1],D=len(seen),end=d['end'],seed=d['run_seed'],thoughts=[s.get('thought') for s in d['steps'] if isinstance(s,dict)] if d.get('steps') else []))
    return out
for agent in ['qwen3-8b-nothink','gpt-oss-20b-low']:
    de=rec(agent,'deadend'); print(agent,'deadend ends',collections.Counter(r['end'] for r in de))
    M=[r for r in rec(agent,'maze') if r['D']>0]
    print(' seeds',sorted(collections.Counter(r['seed'] for r in M).items())[:12])
    def gd(rs):
        G=E=0;G3=E3=0;g=collections.Counter();n=collections.Counter()
        for r in rs:
            nd=r['D'] if r['end']!='cap' else r['D']-1
            for d in range(1,nd+1): n[d]+=1
            if r['end']!='cap': g[r['D']]+=1
        return g,n
    g,n=gd(M)
    a=(g[1],n[1]-g[1]); b=(sum(g[d] for d in n if d>=2),sum(n[d]-g[d] for d in n if d>=2))
    print(' p1 vs 2+',a,b,'one-sided rise p',stats.fisher_exact([[b[0],b[1]],[a[0],a[1]]],alternative='greater')[1])
    a3=(sum(g[d] for d in (1,2,3)),sum(n[d] for d in (1,2,3))); b3=(sum(g[d] for d in n if d>=4),sum(n[d] for d in n if d>=4))
    print(' p1-3 vs 4+',a3,b3,'one-sided rise p',stats.fisher_exact([[b3[0],b3[1]-b3[0]],[a3[0],a3[1]-a3[0]]],alternative='greater')[1])
    if agent.startswith('qwen'):
        M2=[r for r in M if r['end']!='cap']; g2,n2=gd(M2); G=sum(g2.values());E=sum(n2.values());h=G/E
        bins=[(1,3),(4,10),(11,30),(31,999)]
        tb=[(sum(g2[d] for d in range(x,y+1)),sum(n2[d] for d in range(x,y+1))) for x,y in bins]; tb=[t for t in tb if t[1]]
        chi=sum((gg-nn*h)**2/(nn*h)+((nn-gg)-nn*(1-h))**2/(nn*(1-h)) for gg,nn in tb)
        print(' w/o cap: bins',tb,'pearson p',stats.chi2.sf(chi,len(tb)-1))
    # seed blocks: group by seed // something
    seeds=sorted(set(r['seed'] for r in M))
    print(' n distinct seeds',len(seeds), seeds[:5])
