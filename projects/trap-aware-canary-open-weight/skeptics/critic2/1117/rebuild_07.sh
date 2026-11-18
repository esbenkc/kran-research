# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json, pprint
from pathlib import Path
a=json.loads(Path('kran-research/projects/trap-aware-canary-open-weight/results/analysis.json').read_text())['main']
print('main keys',a.keys())
for k,v in a.items():
    if k!='cells':
        print('\n',k); pprint.pp(v, width=120)
PY
