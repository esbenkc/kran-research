# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json,random,math
from pathlib import Path
rows=[json.loads(l) for l in Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/scores.jsonl').read_text().splitlines()]
rng=random.Random(0)
def pct(vals,p):
 vals=sorted(vals); x=(len(vals)-1)*p/100; lo=math.floor(x); hi=math.ceil(x); return vals[lo] if lo==hi else vals[lo]*(hi-x)+vals[hi]*(x-lo)
def rate(rs,key): return sum(bool(r.get(key)) for r in rs)/len(rs)
def D(key):
 return (rate([r for r in rows if r['cell']=='neutral-cued'],key)-rate([r for r in rows if r['cell']=='warning-cued'],key))-(rate([r for r in rows if r['cell']=='neutral-none'],key)-rate([r for r in rows if r['cell']=='warning-none'],key))
for key in ['fit','fit_tok']:
 vals=[]
 by={(cell,t):[r for r in rows if r['cell']==cell and r['template']==t] for cell in ['neutral-cued','warning-cued','neutral-none','warning-none'] for t in set(r['template'] for r in rows)}
 for _ in range(10000):
   sample={}
   for cell in ['neutral-cued','warning-cued','neutral-none','warning-none']:
     s=[]
     for t in sorted(set(r['template'] for r in rows)):
       g=by[(cell,t)]
       for __ in range(len(g)): s.append(rng.choice(g))
     sample[cell]=s
   vals.append((rate(sample['neutral-cued'],key)-rate(sample['warning-cued'],key))-(rate(sample['neutral-none'],key)-rate(sample['warning-none'],key)))
 print(key, round(D(key),4), round(pct(vals,2.5),4), round(pct(vals,97.5),4))
PY
