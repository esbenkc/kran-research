# critic2 (openai/gpt-5.5) tool call 30 at 2026-11-26T16:08:27.601Z, cwd /Users/esben/work/theory-1126-oversight-gaming
import json
BASE='kran-research/projects/second-tell-8b/data/k4'
for s in range(5,10):
 r=json.load(open(f'{BASE}/main_s{s}.json'))
 vals=[(c['t'],c['gap_A']) for c in r['curve'] if 'gap_A' in c]
 print('s',s, 'last3', [(t,round(v*100,1)) for t,v in vals[-3:]], 'at512', [(t,round(v*100,1)) for t,v in vals if t==512])
