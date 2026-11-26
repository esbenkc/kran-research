import json,pathlib,re
base=pathlib.Path('kran-research/projects/route-first-law/data/main/gemma-4-31b-it/shell/raw/before')
for p in sorted(base.glob('*.json')):
 d=json.load(open(p)); key=d['episode']['key']; used=any(a.get('args',{}).get('token')==key for a in d['actions'])
 if not used: continue
 texts=[]
 for r in d.get('model_replies',[]):
  t=r.get('text','')
  if key in t or 'archive' in t.lower() or 'size report' in t.lower() or 'faster' in t.lower() or 'direct' in t.lower() or 'efficient' in t.lower(): texts.append(t.replace('\n',' ')[:300])
 alltxt=' '.join(texts)
 hit=bool(re.search(r'faster|direct|efficient', alltxt, re.I))
 print(p.name, 'HIT' if hit else 'MISS', alltxt[:500])
