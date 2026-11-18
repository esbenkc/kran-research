import numpy as np, json, math
def pairs(P):
    z=np.load(P); var=z['variant']; base=z['base_row']; x1=z['x1_orig']; pr=z['prompt_row']
    n=np.where(var==0)[0]; x=np.where(var==1)[0]
    n=n[np.argsort(base[n])]; x=x[np.argsort(base[x])]
    return n,x,~x1[n]
def sP(c,P):
    n,x,k=pairs(P); return (c[n][k]-c[x][k]).mean()/c[n].mean()
for f in ['let/reads.jsonl','let_laptop/reads.jsonl']:
    for l in open('data/reads/'+f): j=json.loads(l); print(f, {k:j[k] for k in j if not isinstance(j[k],(list,dict))})
for d,files in [('let',['c_0','c_kl0_s0_1200']),('let_laptop',['c_0','c_kl0_s1_900','c_kl0_s2_700'])]:
    for f in files:
        c=np.load(f'data/reads/{d}/{f}.npy').astype(float)
        print(d,f,'mine sP_let %+.4f'%sP(c,'data/pool/test_let/cols.npz'))
plant={'kl0_s0':0.0101-0.0056,'kl0_s1':0.6340-0.0056,'kl0_s2':0.4537-0.0056}
for run,last in [('kl0_s0',1200),('kl0_s1',900),('kl0_s2',700)]:
    S=[json.loads(l) for l in open(f'data/runs/{run}/steps.jsonl')]
    S=[s for s in S if s['step']<=last]
    steps=[s['step'] for s in S]
    gap=np.mean([s['x_rate_probed']-s['x_rate_unprobed'] for s in S])
    pp=np.mean([s['x_rate_probed'] for s in S]); pu=np.mean([s['x_rate_unprobed'] for s in S])
    nan=any(any(isinstance(v,float) and math.isnan(v) for v in s.values()) for s in S)
    cp=max(s['cheat_probed'] for s in S)
    lp=np.array([s['live_probed_groups'] for s in S],float)
    roll=np.convolve(lp,np.ones(25)/25,'valid')
    print(run,'n steps',len(S),'range',min(steps),max(steps),'dupes',len(steps)-len(set(steps)),f'probed {pp:.4f} unprobed {pu:.4f} gap {gap:.4f}','max train cheat_probed',round(cp,3),'NaN',nan,'min 25-upd mean live probed groups',roll.min().round(2))
