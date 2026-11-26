import json,pathlib,re
for name in ['004','008','011','012','017','019','021']:
 d=json.load(open(f'kran-research/projects/route-first-law/data/main/gemma-4-31b-it/shell/raw/before/{name}.json'))
 print('---',name)
 for r in d['model_replies']:
  t=r.get('reasoning','')
  if 'observation provides' in t.lower() or 'route' in t.lower() or 'archive' in t.lower() or 'token' in t.lower():
   print(t[:1000])
