# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json
from pathlib import Path
rows=[json.loads(l) for l in Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/scores.jsonl').read_text().splitlines()]
for cells in [('warning-cued',),('neutral-cued',),('warning-cued','neutral-cued')]:
 rs=[r for r in rows if r['cell'] in cells]
 print(cells, 'hidden_rt all', sum(r['hidden_rt'] for r in rs),'/',len(rs),'hidden_rt among fit', sum(r['hidden_rt'] for r in rs if r['fit']),'/',sum(r['fit'] for r in rs),'markup among fit',sum(r['markup_rt'] for r in rs if r['fit']))
PY
