# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/raw/warning-cued/030.json').read_text())
# print top keys and maybe steps summaries
print(r.keys())
for i,s in enumerate(r.get('steps', [])):
    print('STEP', i, s.keys())
    for k,v in s.items():
        if isinstance(v,str) and ('666221' in v or 'trap' in v.lower() or 'display:none' in v or 'footer' in v.lower()):
            print(k, v[:1000].replace('\n',' '))
PY
