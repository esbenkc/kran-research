
import json
for r in ['ao_s0','ao_s1','ao_s2','p10_s0','p10_s1','p10_s2','p100_s0']:
    tl=json.load(open(f'data/runs/{r}.json'))['train_log']
    print(r, len(tl), 'pen==0:',sum(1 for x in tl if x[2]==0), 'last update', tl[-1][0])
