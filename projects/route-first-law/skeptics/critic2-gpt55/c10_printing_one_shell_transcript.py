import json,pathlib; d=json.load(open('kran-research/projects/route-first-law/data/main/gemma-4-31b-it/shell/raw/before/000.json'));
for r in d['model_replies']:
 print('STEP',r['step']); print(r.get('reasoning','')); print(r.get('text','')); print('---')
