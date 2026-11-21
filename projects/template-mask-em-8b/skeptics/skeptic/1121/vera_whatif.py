"""Vera: what can seed 2 still do to the frozen status, given seeds 0 and 1 from vera_check_out.json?
Grid over seed-2 cut(B), habit(B) (and cut(E) = its 2-seed mean, or a worst/best case), applying the
FROZEN RULE status (PREREG 'Status') with the t(2) = 4.303 seed-level interval.
Usage: vera_whatif.py vera_check_out.json
"""
import json, sys
import numpy as np

T2 = 4.302652729911275


def tint(x):
    x = np.asarray(x, float)
    m, h = x.mean(), T2 * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def status(cb, hb, ce):
    c, h = tint(cb), tint(hb)
    if c[2] < 0.5 or h[2] < 0.8:
        return 'REJECTED'
    if c[0] >= 0.5 and h[0] >= 0.8 and c[1] > 0 and c[0] > np.mean(ce):
        return 'CONFIRMED'
    return 'UNCONFIRMED'


d = json.load(open(sys.argv[1]))
ps = d['per_seed']
seeds = sorted({k.split('|')[1] for k in ps if k.startswith('B|')})[:2]
cb = [ps[f'B|{s}']['cut'] for s in seeds]
hb = [ps[f'B|{s}']['habit'] for s in seeds]
ce = [ps[f'E|{s}']['cut'] for s in seeds]
print('seeds', seeds, 'cut(B)', cb, 'habit(B)', hb, 'cut(E)', ce)
grid = np.round(np.arange(-1.0, 2.0001, 0.01), 2)
out = {}
for ce2 in (min(ce), float(np.mean(ce)), max(ce), 1.5):
    counts = {}
    reach = {}
    for c2 in grid:
        for h2 in grid:
            s = status(cb + [c2], hb + [h2], ce + [ce2])
            counts[s] = counts.get(s, 0) + 1
            reach.setdefault(s, []).append((float(c2), float(h2)))
    summ = {}
    for s, pts in reach.items():
        a = np.array(pts)
        summ[s] = dict(n=len(pts), cut2=[a[:, 0].min(), a[:, 0].max()], habit2=[a[:, 1].min(), a[:, 1].max()])
    out[f'cutE_seed2={ce2:.3f}'] = summ
    print(f'seed-2 cut(E) = {ce2:.3f}:', json.dumps(summ, default=float))
json.dump(dict(seeds=seeds, cutB=cb, habitB=hb, cutE=ce, grid='seed-2 cut, habit in [-1, 2] step 0.01', out=out),
          open('vera_whatif_out.json', 'w'), indent=1, default=float)
