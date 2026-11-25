"""Robustness of the REJECTED-vs-UNCONFIRMED edge: L vs best null median miss, per seed and by cell bootstrap,
under both readings of 'a failed cell is a miss of 100' (all predictors vs the law only); plus shuffled-q null."""
import json, numpy as np
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
out, cells, prim, train = score(P, verbose=False)
pj = json.load(open(P + '/predictions.json'))['cells']
keys = sorted(prim); res = {}
def misses(reading, seed=None):
    D = {n: [] for n in ['L', 'N1', 'N2', 'N3', 'N4']}
    for k in keys:
        v = prim[k]; c = pj[f"{k[0]}x{k[1]}x{k[2]}r{k[3]}|nb{int(k[4])}_q{k[5]:g}_c{k[6]:g}"]
        em = v['e_seed'][seed] if seed is not None else np.mean(list(v['e_seed'].values()))
        for n in D:
            m = abs(c['e_' + n] - em) * 100
            if not v['ok'] and (reading == 'all' or n == 'L'): m = 100.0
            D[n].append(m)
    return {n: np.array(x) for n, x in D.items()}
rng = np.random.default_rng(0)
for reading in ('all', 'law_only'):
    D = misses(reading)
    med = {n: float(np.median(x)) for n, x in D.items()}
    best = min(['N1', 'N2', 'N3', 'N4'], key=lambda n: med[n])
    per_seed = {s: {n: float(np.median(x)) for n, x in misses(reading, s).items()} for s in (0, 1, 2)}
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(keys), len(keys))
        bs.append(np.median(D['L'][i]) - min(np.median(D[n][i]) for n in ['N1', 'N2', 'N3', 'N4']))
    bs = np.array(bs)
    res[reading] = dict(medians=med, best_null=best, per_seed=per_seed, P_L_ge_best=float(np.mean(bs >= 0)), diff=float(med['L'] - med[best]))
    print(reading, {n: round(m, 2) for n, m in med.items()}, 'best', best, 'L-best %.2f' % (med['L'] - med[best]),
          'bootstrap P(L>=best null) %.2f' % np.mean(bs >= 0))
    for s, m in per_seed.items(): print('   seed', s, {n: round(x, 2) for n, x in m.items()})
# converged only
ok = [k for k in keys if prim[k]['ok']]
print('converged-only medians', {n: round(float(np.median([abs(pj[f"{k[0]}x{k[1]}x{k[2]}r{k[3]}|nb{int(k[4])}_q{k[5]:g}_c{k[6]:g}"]['e_' + n] - np.mean(list(prim[k]['e_seed'].values()))) * 100 for k in ok])), 2) for n in ['L', 'N1', 'N2', 'N3', 'N4']})
sn = shuffle_null(P, n_perm=300)
print('shuffled-q null (law share):', sn)
res['shuffle_null'] = sn
json.dump(res, open('vera_robust.json', 'w'), indent=1)
