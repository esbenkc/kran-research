
import json
o=json.load(open('/tmp/critic2_rebuild_1124.json'))
for r,v in o.items(): print(r, '%.1f%% less'%(100*(1-v['push']/v['cpush'])))
d=json.load(open('data/runs/ao_s0.json')); print(list(d.keys())[:20]); tl=d.get('train_log'); print(type(tl), len(tl) if tl else None); print(tl[0] if tl else None)
