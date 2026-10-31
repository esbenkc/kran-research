"""Vera: s7 prediction outcome with 90% CIs (models as clusters, runs resampled within each model x decoy x arm cell)."""
import json, numpy as np
from collections import defaultdict
R=[json.loads(l) for l in open('vera_counted_scores.jsonl')]
C=defaultdict(lambda: defaultdict(lambda: ([],[])))
for r in R:
    c=C[(r['model'],r['decoy'])][r['prompt']]; c[0].append(bool(r['says_v13'])); c[1].append(bool(r['trip']))
C={k:{a:(np.array(v[0]),np.array(v[1])) for a,v in d.items()} for k,d in C.items()}
models=sorted({k[0] for k in C})
def stats(pick):
    sr=[];td=[];ts=[];
    for (m,d) in pick:
        on,off=C[(m,d)]['on'],C[(m,d)]['off']
        sr.append(on[0].mean()-off[0].mean()); td.append(off[1].mean()-on[1].mean())
    allS=np.concatenate([C[k]['on'][1][C[k]['on'][0]] for k in pick]+[C[k]['off'][1][C[k]['off'][0]] for k in pick])
    return np.mean(sr), np.mean(td), allS.mean()
pt=stats(list(C))
rng=np.random.default_rng(31); bs=[]
for _ in range(4000):
    ms=rng.choice(models,len(models)); pick=[]
    save={}
    for m in ms:
        for d in ('obvious','subtle'):
            k=(m,d); pick.append(k)
    # runs within cell
    bak={k:C[k] for k in C}
    for k in set(pick):
        C[k]={a:(lambda s,t,i:(s[i],t[i]))(*bak[k][a], rng.integers(0,len(bak[k][a][0]),len(bak[k][a][0]))) for a in bak[k]}
    bs.append(stats(pick)); C.update(bak)
bs=np.array(bs); q=np.quantile(bs,[0.05,0.95],axis=0)
for i,(n,th,ok) in enumerate((('(a) say rise','>= 0.40',lambda x:x>=0.40),('(b) trip drop','<= 0.10',lambda x:x<=0.10),('(c) trip|says','>= 0.50',lambda x:x>=0.50))):
    print(f"{n}: {pt[i]:.3f} 90% CI [{q[0,i]:.3f}, {q[1,i]:.3f}] pred {th}: point {'meets' if ok(pt[i]) else 'misses'}")
