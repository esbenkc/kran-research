# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json, random, statistics, math
from pathlib import Path
rows=[json.loads(l) for l in Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/scores.jsonl').read_text().splitlines()]
rng=random.Random(0)
def pct(vals,p):
    vals=sorted(vals)
    # nearest-rank? use numpy-like linear
    x=(len(vals)-1)*p/100
    lo=math.floor(x); hi=math.ceil(x)
    if lo==hi: return vals[lo]
    return vals[lo]*(hi-x)+vals[hi]*(x-lo)
def boot(cell, key, B=10000):
    groups=[]
    for t in sorted(set(r['template'] for r in rows if r['cell']==cell)):
        g=[r for r in rows if r['cell']==cell and r['template']==t]
        groups.append(g)
    vals=[]
    for _ in range(B):
        k=n=0
        for g in groups:
            for __ in range(len(g)):
                r=rng.choice(g); k += bool(r.get(key)); n += 1
        vals.append(k/n)
    return pct(vals,2.5), pct(vals,97.5)
for cell,key in [('warning-cued','fit'),('neutral-cued','fit'),('warning-none','fit'),('neutral-none','fit'),('warning-cued','fit_tok'),('neutral-cued','fit_tok'),('warning-none','fit_tok'),('neutral-none','fit_tok')]:
    k=sum(1 for r in rows if r['cell']==cell and r.get(key)); n=sum(1 for r in rows if r['cell']==cell)
    ci=boot(cell,key)
    print(cell,key,k,n,round(ci[0],4),round(ci[1],4))
PY
