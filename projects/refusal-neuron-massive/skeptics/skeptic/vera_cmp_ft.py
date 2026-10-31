"""Vera: compare my own-forward first-token refusal log-odds (CPU fp32, own edit code) with Bruno's saved scores
(MPS bf16) on the first 50 TEST prompts, per arm, and the arm-vs-base drop in each."""
import json, numpy as np, os
D = '/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/runs/logit/test_harmful'
mine = lambda a: np.array([json.loads(l)['logodds_R'] for l in open(f'own/ft_{a}_0_50.jsonl')])
his = lambda a: np.array(json.load(open(f'{D}/{a}.json'))['score'])[:50]
mb, hb = mine('base'), his('base')
for a, h in [('base', 'base'), ('A', 'A'), ('B', 'B'), ('C', 'C'), ('A_tok0', 'A_tok0'), ('N_5441_1', 'N5441pos')]:
    if not os.path.exists(f'own/ft_{a}_0_50.jsonl'):
        continue
    m, b = mine(a), his(h)
    print(f'{a:9s} corr {np.corrcoef(m, b)[0,1]:.4f} mean|diff| {np.abs(m-b).mean():.3f} | drop vs base: mine {(mb-m).mean():+.3f} his {(hb-b).mean():+.3f}')
