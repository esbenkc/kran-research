# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json
from pathlib import Path
a=json.loads(Path('kran-research/projects/trap-aware-canary-open-weight/results/analysis.json').read_text())
# find keys around status/deterrence print compact top-level and selected nested
print('top', sorted(a.keys()))
for key in ['main','status','deterrence','posthoc','competence','parent']:
    if key in a: print(key, type(a[key]).__name__)
# recursively find dicts with deterrence maybe
def walk(o,path=''):
    if isinstance(o,dict):
        for k,v in o.items():
            if 'det' in k.lower() or k in ['status','status_blockers','primary','headline','competence']:
                print(path+'/'+k, v if not isinstance(v,(dict,list)) else type(v).__name__)
            walk(v,path+'/'+str(k))
walk(a)
PY
