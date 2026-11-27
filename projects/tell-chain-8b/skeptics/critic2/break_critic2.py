"""critic2 break attempts: switch-threshold sensitivity (0.05/0.15/0.20) and leave-one-seed-out upper ends."""
import json, math, sys
import numpy as np
from scipy import stats
P = sys.argv[1]; base = json.load(open(f'{P}/data/base_px.json')); S = [5, 6, 7, 8, 9]
def T(name, cue, thr):
    r = json.load(open(f'{P}/data/chain/{name}.json')); s = r['seed']
    gb = np.mean(base['p_base_off']) - np.mean(base[f'p_base_{s}{cue}'])
    D = [(c['t'], np.mean(c['p_off']) - np.mean(c[f'p_{cue}']) - gb) for c in sorted(r['curve'], key=lambda c: c['t'])]
    return next((D[i][0] for i in range(len(D) - 1) if D[i][1] >= thr and D[i + 1][1] >= thr), None)
def ci(lr):
    lr = np.array(lr); m = lr.mean(); h = stats.t.ppf(.975, len(lr) - 1) * lr.std(ddof=1) / math.sqrt(len(lr))
    return round(math.exp(m), 3), round(math.exp(m - h), 3), round(math.exp(m + h), 3)
for thr in (0.05, 0.10, 0.15, 0.20):
    row = {}
    for ref in ('ctrl_D', 'fresh_D'):
        pairs = [(T(f'main_D_s{s}', 'D', thr), T(f'{ref}_s{s}', 'D', thr)) for s in S]
        if any(a is None or b is None for a, b in pairs):
            row[ref] = ('uncrossed in curve', pairs); continue
        row[ref] = ci([math.log(max(a, 8) / max(b, 8)) for a, b in pairs])
    print('thr', thr, row)
for ref in ('ctrl_D', 'fresh_D'):
    lr = [math.log(max(T(f'main_D_s{s}', 'D', .1), 8) / max(T(f'{ref}_s{s}', 'D', .1), 8)) for s in S]
    print('LOO', ref, [ci(lr[:i] + lr[i + 1:])[2] for i in range(5)], '< 0.55 all:', all(ci(lr[:i] + lr[i + 1:])[2] < .55 for i in range(5)))
