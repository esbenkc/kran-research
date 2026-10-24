# Owain: per-city log-prob margin (right country minus wrong country) in each model state, using metrics.pairs().
import json, sys, numpy as np
sys.path.insert(0, '.')
from common import ACTS
from metrics import pairs, EV_CITIES
P = pairs()
out = {}
for st in ['clean'] + [f'a__gap__lam10000__seed{s}' for s in range(3)] + [f'b__gap300__lam10000__seed{s}' for s in range(3)] + ['a__gap__lam0__seed0']:
    lp, off = np.load(ACTS / st / 'lp.npy'), np.load(ACTS / st / 'off.npy')
    m = {}
    for c, (r, w, p, cr, cw) in zip(EV_CITIES, P):
        m[c] = float(lp[off[r] + p - 1: off[r] + cr].sum() - lp[off[w] + p - 1: off[w] + cw].sum())
    out[st] = m
json.dump(out, open(sys.argv[1], 'w'))
print('ok', len(EV_CITIES))
