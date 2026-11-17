# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/scores.jsonl').read_text().splitlines()[0])
print(sorted(r.keys()))
for idx,line in enumerate(Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/scores.jsonl').read_text().splitlines()):
    r=json.loads(line)
    if r['cell']=='warning-cued' and r['fit']:
        print(idx, {k:r.get(k) for k in sorted(r.keys()) if 'hidden' in k or 'know' in k or 'markup' in k or 'trap' in k or k in ['trace','fit','i']})
        break
PY
