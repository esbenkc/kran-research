# critic2 (openai/gpt-5.5) tool call 27 at 2026-11-26T16:08:04.616Z, cwd /Users/esben/work/theory-1126-oversight-gaming
import json, statistics
p='kran-research/projects/second-tell-8b/data/subspace_k4.json'
d=json.load(open(p))
print('arm_mean_angle', {k:round(v,3) for k,v in d['arm_mean_angle'].items()})
print('random', {k:round(v,3) for k,v in d['random_reference'].items()})
print('clause', d['clause'])
# own first-stage overlap? inspect tails keys for own_stage angle
for name in ['main_s5','shuf_s5','coin_s5','fresh_s5']:
 print(name, d['tails'][name].keys())
 print(d['tails'][name])
