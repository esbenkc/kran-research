import json,pathlib,re
base=pathlib.Path('kran-research/projects/route-first-law/data/main/gemma-4-31b-it/shell/raw/before')
patterns=[r'faster|direct|efficient']
miss=[]; hit=0; size=0
for p in sorted(base.glob('*.json')):
 d=json.load(open(p)); key=d['episode']['key']; used=any(a.get('args',{}).get('token')==key for a in d['actions'])
 if not used: continue
 txt='\n'.join([r.get('reasoning','')+'\n'+r.get('text','') for r in d.get('model_replies',[])])
 if re.search(r'size[- ]report|size report', txt, re.I): size+=1
 if re.search(r'faster|direct|efficient', txt, re.I): hit+=1
 else: miss.append(p.name)
print('key uses', hit+len(miss),'size report mention', size,'fast/direct/efficient', hit,'misses',miss)
