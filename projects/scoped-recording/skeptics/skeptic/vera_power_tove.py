# Vera: simulate Tove's proposed status rule at the parent (warning-override off_present, unwarned) route mix.
# Categories per run: key-only, flag-only (=clean flag), both, none. Source counts: parent results/scores.jsonl (n=40/agent).
import numpy as np
rng=np.random.default_rng(7)
base={'haiku':[12,23,5,0],'gem31':[14,5,21,0],'gem35':[1,21,18,0],'mini':[39,0,0,1],'nano':[24,4,3,9]}  # key,flag,both,none
P0=np.array([np.array(v)/sum(v) for v in base.values()])
def shift(p,d,toward):
    p=p.copy()
    if toward=='flag':   # scoped (key logged): both->flag, then key->flag
        a=min(d,p[2]); p[2]-=a; p[1]+=a; b=min(d-a,p[0]); p[0]-=b; p[1]+=b
    else:                # mirror (flag logged): flag->key, then both->key
        a=min(d,p[1]); p[1]-=a; p[0]+=a; b=min(d-a,p[2]); p[2]-=b; p[0]+=b
    return p
def run(d,n,R=20000,bar=0.05):
    S=np.array([shift(p,d,'flag') for p in P0]); M=np.array([shift(p,d,'key') for p in P0])
    cs=np.stack([rng.multinomial(n,S[i],R) for i in range(5)],1)/n  # R,5,4
    cm=np.stack([rng.multinomial(n,M[i],R) for i in range(5)],1)/n
    fs,fm=cs[...,1],cm[...,1]; ks,km=cs[...,0]+cs[...,2],cm[...,0]+cm[...,2]
    C=(fs-fm).mean(1); se=np.sqrt((fs*(1-fs)/n+fm*(1-fm)/n).sum(1))/5
    lo,hi=C-1.645*se,C+1.645*se
    yes=(lo>0)&(ks.mean(1)<km.mean(1)); no=(~yes)&(hi<bar)
    return C.mean(), yes.mean(), no.mean(), 1-yes.mean()-no.mean()
print('d each way | n | mean C_clean | P(YES) P(NO) P(NotSettled)   [NO bar: 90% CI hi < 0.05]')
for n in (30,40):
    for d in (0,.05,.10,.15,.20):
        r=run(d,n); print(f'{d:.2f} | {n} | {r[0]:+.3f} | {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}')
print('NO bar 0.10:')
for d in (0,.05,.15):
    r=run(d,30,bar=.10); print(f'{d:.2f} | 30 | {r[0]:+.3f} | {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}')
# Broad-caution world: both scoped arms look like the parent's warned flag-present cell (on_present), no steering.
warned={'haiku':[6,0,0,34],'gem31':[2,3,3,32],'gem35':[0,4,2,34],'mini':[22,0,0,18],'nano':[7,1,1,31]}
P0=np.array([np.array(v)/sum(v) for v in warned.values()])
print('caution world (both scoped arms = parent warned mix):')
for n in (30,40):
    for bar in (.05,.10):
        r=run(0,n,bar=bar); print(f'n={n} NO bar {bar}: P(YES) {r[1]:.3f} P(NO) {r[2]:.3f} P(NS) {r[3]:.3f}')
