import json, pathlib; base=pathlib.Path('kran-research/projects/route-first-law/data/main');
for p in [base/'gemma-4-26b-a4b-it/web/raw/before/000.json', base/'gemma-4-31b-it/shell/raw/before/000.json']:
 d=json.load(open(p)); print(p); print(d.keys()); print('web fields', {k:d.get(k) for k in ['model','view','harness_followed','exposed','correct','seed_block','arm','run_id']}); print('site keys', d.get('site',{}).keys()); print('shell fields', {k:d.get(k) for k in ['decoy_log','seed_block','model','route_place']}); print('actions0', d.get('actions',[{}])[:1]); print()
