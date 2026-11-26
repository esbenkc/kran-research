# critic2 (openai/gpt-5.5) tool call 22 at 2026-11-26T16:07:39.306Z, cwd /Users/esben/work/theory-1126-oversight-gaming
import json, math, glob, statistics, os
BASE='kran-research/projects/second-tell-8b/data/k4'
# own T_B from curve: first t where D_B>=0.10 and next check D_B>=0.10
runs={}
for path in sorted(glob.glob(BASE+'/*.json')):
    r=json.load(open(path)); name=r['name']; curve=r['curve']
    T=None
    for c,n in zip(curve, curve[1:]):
        if c['D_B'] >= 0.10 and n['D_B'] >= 0.10:
            T=c['t']; break
    if T is None: T=1536
    runs[name]={'arm':r['arm'],'seed':r['seed'],'T_calc':T,'T_file':r['T_B'],'censored':r['censored'],'D0':curve[0]['D_B']}
# print T table ordered like post
arms=['main','fresh','shuf','coin']
print('T table from raw curves:')
for arm in arms:
    vals=[]
    for s in range(5,10):
        name=f'{arm}_s{s}'
        vals.append(runs[name]['T_calc'])
        assert runs[name]['T_calc']==runs[name]['T_file'], (name,runs[name])
    geo=math.exp(sum(math.log(max(x,8)) for x in vals)/len(vals))
    print(arm, vals, 'geo', round(geo,3))
print('censored', sum(v['censored'] for v in runs.values()))
# ratios and t intervals df=4 tcrit
T={arm:[runs[f'{arm}_s{s}']['T_calc'] for s in range(5,10)] for arm in arms}
tcrit=2.7764451051977987
print('ratios vs controls:')
for ref in ['fresh','shuf','coin']:
    lrs=[math.log(max(T['main'][i],8)/max(T[ref][i],8)) for i in range(5)]
    mean=sum(lrs)/5
    sd=statistics.stdev(lrs)
    se=sd/math.sqrt(5)
    lo=mean-tcrit*se; hi=mean+tcrit*se
    print(ref, 'R', round(math.exp(mean),6), 'CI', (round(math.exp(lo),6), round(math.exp(hi),6)), 'per_seed', [round(math.exp(x),3) for x in lrs])
print('D0 max abs by arm:')
for arm in arms:
    vals=[runs[f'{arm}_s{s}']['D0'] for s in range(5,10)]
    print(arm, max(abs(x) for x in vals), [round(x,4) for x in vals])
# A retention main ranges
ret=[]
for s in range(5,10):
    r=json.load(open(f'{BASE}/main_s{s}.json'))
    # curve entries with gap_A t=0 and last gap_A
    gap0=r['curve'][0]['gap_A']
    gaps=[c.get('gap_A') for c in r['curve'] if 'gap_A' in c]
    ret.append((s,gap0,gaps[-1], [c['t'] for c in r['curve'] if 'gap_A' in c][-1]))
print('A retention', [(s, round(a*100,1), round(b*100,1), t) for s,a,b,t in ret])
