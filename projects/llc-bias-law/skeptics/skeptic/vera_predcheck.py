"""Recompute every blind/seen e_L, e_N2 in predictions.json with my own law code and my own fit (a, b from raw blocks)."""
import json, numpy as np
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
pj = json.load(open(P + '/predictions.json'))
a, b = pj['fit']['L']['a'], pj['fit']['L']['b']
dev = []; devN2 = []; n_prim = 0
a2, b2 = 0.0101, 0.511
for k, c in pj['cells'].items():
    M, N, H, r = [int(t) for t in c['case'].replace('r', 'x').split('x')]
    e = law_pred((M, N, H, r), c['nbeta'], c['q'], c['eps'], a, b)
    dev.append(abs(e - c['e_L']) * 100)
    lam, reg, lam0 = lam_parts(M, N, H, r)
    assert abs(lam - c['lam']) < 1e-9 and abs(lam0 - c['lam0']) < 1e-9
    assert abs(c['gamma'] - np.sqrt(c['nbeta'] * 0.5 / c['q'])) < 1e-9 and abs(c['eps'] - c['c'] / (c['nbeta'] * 2 + c['gamma'])) < 1e-12
    if c['group'] == 'blind' and c['primary']: n_prim += 1
print('cells', len(pj['cells']), 'primary blind', n_prim, 'max |my e_L - theirs| (pts)', max(dev))
