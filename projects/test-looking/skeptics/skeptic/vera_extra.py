# Vera extra checks: scout clause 3 (pooled drop CI), cap hits by arm, p_test per agent, per-agent exact tests.
import json, numpy as np, collections, glob, sys
from math import comb
rows=[json.loads(l) for l in open('vera_rebuild_runs.jsonl')]
raw={}
for f in glob.glob('../../kran-research/projects/test-looking/raw/*/*/*.jsonl'):
    r=json.loads(open(f).readline()); raw[(r['model'],r['arm'],r['seed'])]=r
S=range(1000,1040)
P=np.array([r['use'] for r in rows if r['arm']=='prod' and r['seed'] in S]); T=np.array([r['use'] for r in rows if r['arm']=='test' and r['seed'] in S])
rng=np.random.default_rng(7); bs=[rng.choice(P,len(P)).mean()-rng.choice(T,len(T)).mean() for _ in range(4000)]
print('clause2/3 pooled drop pts', round(100*(P.mean()-T.mean()),2), 'CI', [round(100*x,1) for x in np.percentile(bs,[2.5,97.5])])
cap=collections.Counter(); n=collections.Counter()
for (m,a,s),r in raw.items():
    n[a]+=1; cap[a]+=int(r['steps_used']>=r['max_steps'])
print('cap hits by arm', {a:f'{cap[a]}/{n[a]}' for a in n})
def fisher(a,n1,c,n2):
    K=a+c; N=n1+n2
    pr=lambda x: comb(n1,x)*comb(n2,K-x)/comb(N,K)
    p0=pr(a); return sum(pr(x) for x in range(max(0,K-n2),min(K,n1)+1) if pr(x)<=p0*1.0000001)
for m in sorted({r['model'] for r in rows}):
    p=[r for r in rows if r['model']==m and r['arm']=='prod' and r['seed'] in S]; t=[r for r in rows if r['model']==m and r['arm']=='test' and r['seed'] in S]
    a=sum(r['use'] for r in p); c=sum(r['use'] for r in t)
    pp=[r['p_test'] for r in p if r['p_test'] is not None]; tp=[r['p_test'] for r in t if r['p_test'] is not None]
    print(m, f'use {a}/40 -> {c}/40 fisher p={fisher(a,40,c,40):.3f}', f'p_test {np.mean(pp) if pp else None:.1f} (n{len(pp)}) -> {np.mean(tp) if tp else float("nan"):.1f} (n{len(tp)})')
