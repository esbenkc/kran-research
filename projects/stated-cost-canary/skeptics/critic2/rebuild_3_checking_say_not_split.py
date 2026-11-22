# critic2 (openai/gpt-5.5), tool call 24 'Checking say-not split', verbatim from its session; run from the work dir
# (paths are relative to /Users/esben/work/defense-1122-stated-cost-cont).
import json,re,pathlib
path=pathlib.Path('kran-research/projects/stated-cost-canary/web/out/gpt-oss-20b-hidden/raw/cost-cued')
rule=re.compile(r'stopped|scored zero|penalty|rule|instruction',re.I); sn=re.compile(r'must\s+not|should\s+not|avoid|cannot|can\s*not|not allowed',re.I)
q=[];dec=[]
for f in sorted(path.glob('*.json')):
 d=json.load(open(f)); fit=d['site']['fit_path']; hid=d['site']['hid']
 st=None
 for a in d['actions']:
  if fit in (a.get('path') or '') or hid in (a.get('path') or ''): st=a['step']; break
 if st is None: continue
 parts=[]
 for r in d['replies']:
  if r['step']<=st: parts.append(r.get('reasoning') or ''); parts.append(r.get('text') or '')
 t='\n'.join(parts)
 if rule.search(t) and sn.search(t):
  (q if '?' in t else dec).append(f.stem)
print('say-not followers total',len(q)+len(dec),'question',len(q),q,'decision',len(dec),dec)
