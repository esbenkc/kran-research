# P(Not settled) under the frozen YES/NO bars, from the parent's dev curves (numeric-chameleon-rl, pull runs)
import json
P='/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/data/runs/'
a={r['step']:r for r in json.load(open(P+'r50_s0_lr0.0001_kl1.json'))['dev_curve']}
b={r['step']:r for r in json.load(open(P+'r50_s1_lr0.0001_kl1_at8192.json'))['dev_curve']}
steps=sorted(set(a)&set(b))
def cls(g): return 'yes' if g>=0.15 else ('mid' if g>=0.05 else 'low')
out={}
for lo,hi in [(0,8192),(2560,8192),(4096,8192),(4096,5120)]:
    ss=[s for s in steps if lo<=s<=hi]
    pair=[(cls(a[s]['gap_ref']),cls(b[s]['gap_ref'])) for s in ss]
    n=len(ss)
    out[f'{lo}-{hi}']=dict(n=n,
        both_yes=sum(p==('yes','yes') for p in pair)/n,
        any_mid=sum('mid' in p for p in pair)/n,
        split=sum(set(p)=={'yes','low'} for p in pair)/n,
        both_low=sum(p==('low','low') for p in pair)/n,
        bar10_change=sum((0.10<=a[s]['gap_ref']<0.15) or (0.10<=b[s]['gap_ref']<0.15) for s in ss)/n)
# first step each seed crosses 0.15
for nm,c in (('s0',a),('s1',b)):
    out['jump_'+nm]=min(s for s in c if c[s]['gap_ref']>=0.15)
    out['min_gap_after_jump_'+nm]=min(c[s]['gap_ref'] for s in c if s>=out['jump_'+nm] and s<=8192)
print(json.dumps(out,indent=1))
json.dump(out,open('/Users/esben/work/oracle-1106-shaping-watched/swarm-oracle-1106-shaping-watched/skeptic/vera_pns.json','w'),indent=1)
