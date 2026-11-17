# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import csv
rows=list(csv.DictReader(open('kran-research/projects/trap-aware-canary-open-weight/results/parent_api_cells.csv')))
for r in rows:
    if r['agent']=='pooled' and r['hiding']=='cued' and r['event']=='fit' and r['sentence'] in ['warning','neutral']:
        print(r)
PY
