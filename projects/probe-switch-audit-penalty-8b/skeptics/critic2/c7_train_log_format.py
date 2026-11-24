
import json
for r in ['ao_s0','ao_s1','ao_s2']:
    d=json.load(open(f'data/runs/{r}.json')); print(r, d['train_log_format']); tl=d['train_log']
    print(tl[100])
