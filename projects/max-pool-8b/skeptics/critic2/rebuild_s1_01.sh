python3 - <<'PY'
import json, math
from pathlib import Path
p=Path('kran-research/projects/max-pool-8b/data/results/results.json')
r=json.loads(p.read_text())
print('status', r['status'], 'prereg', r['prereg_outcome'])
print('clean gate', r['clean_validity_gate'])
for k,v in r['pooled'].items():
    print('pooled', k, v)
print('fresh', r['fresh_fits'])
print('states keys', list(r['states'])[:10])
for s in ['clean','main__seed0__step150','main__seed1__step150','main__seed2__step150','random__seed0__step150','random__seed1__step150']:
    st=r['states'].get(s)
    if st:
        print('state',s)
        for probe in ['avg18','max18','last18']:
            if probe in st: print(' ',probe, st[probe])
print('diagnostics keys', r['diagnostics'].keys())
print('n', r['n'])
PY
