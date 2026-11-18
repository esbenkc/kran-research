# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import csv
from pathlib import Path
p=Path('kran-research/projects/trap-aware-canary-open-weight/results/parent_api_cells.csv')
rows=list(csv.DictReader(p.open()))
print(rows[0].keys())
for cell in ['W','U']:
    rs=[r for r in rows if r.get('cell')==cell or r.get('variant')==cell]
    print('cell',cell,'nrows',len(rs))
    print(rs[:3])
# infer columns
for col in ['fit','k','n','warned','sentence','hiding','agent','rate','count']:
    if col in rows[0]: print(col, sorted(set(r[col] for r in rows))[:10])
PY
