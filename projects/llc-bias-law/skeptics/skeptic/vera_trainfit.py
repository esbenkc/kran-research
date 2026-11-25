"""Rebuild the train fit (a, b) of law L from raw blocks; compare with predictions.json."""
import json, sys, numpy as np
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
rows = load(P); cells = aggregate(rows)
print('rebuild', REBUILD)
train = {k: v for k, v in cells.items() if k[:4] in TRAIN and k[5] >= 0.1}
print('train cells', len(train), 'not ok', sum(not v['ok'] for v in train.values()))
(a, b), rmse = fit_L(train); print('my L fit a=%.4f b=%.3f rms %.2f pts' % (a, b, rmse))
a2, b2 = fit_N2(train); print('my N2 a=%.4f b=%.3f' % (a2, b2))
n1 = np.mean([np.mean(list(v['e_seed'].values())) for v in train.values()]); print('N1 %.2f%%' % (100*n1))
pj = json.load(open(P + '/predictions.json')); print({k: pj[k] for k in list(pj)[:6] if not isinstance(pj[k], (list, dict)) or len(str(pj[k])) < 400})
