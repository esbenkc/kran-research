# Vera power sim on the SMOKE rates (v1.1). Per model, flag-present cells: k0,f0 (unwarned), k1 (warned) from
# my own vera_smoke_scores.jsonl, shrunk 50/50 toward the pooled rate (5 runs/model/cell is tiny).
# Scenarios for warned flag use f1: flat (f1=f0), +35% of the key drop, and the smoke's own warned flag rate.
# Rule: PREREG s3, b=+0.0725, within-model run bootstrap (Binomial(n, p_hat) is exact for binary resampling).
import json, collections, sys
import numpy as np
rng = np.random.default_rng(2)
c = collections.defaultdict(lambda: [0, 0, 0])
for l in open('vera_smoke_scores.jsonl'):
    d = json.loads(l)
    if d['override'] != 'present':
        continue
    x = c[(d['model'], d['prompt'])]; x[0] += 1; x[1] += d['key']; x[2] += d['flag']
models = sorted({m for m, _ in c})
def rate(m, p, j):
    pool = sum(c[(mm, p)][j] for mm in models) / sum(c[(mm, p)][0] for mm in models)
    return 0.5 * c[(m, p)][j] / c[(m, p)][0] + 0.5 * pool
k0 = np.array([rate(m, 'off', 1) for m in models]); f0 = np.array([rate(m, 'off', 2) for m in models])
k1 = np.array([rate(m, 'on', 1) for m in models]); fs = np.array([rate(m, 'on', 2) for m in models])
print('models', [m[:12] for m in models]); print('k0', k0.round(2), 'k1', k1.round(2), 'f0', f0.round(2), 'f1_smoke', fs.round(2))
b, B, R = 0.0725, 4000, 400
for n in (40, 50):
    for name, f1 in (('flat', f0), ('35% of key drop', np.clip(f0 + 0.35 * (k0 - k1), 0, .99)),
                     ('20% of key drop', np.clip(f0 + 0.20 * (k0 - k1), 0, .99)), ('smoke f1', fs)):
        out = collections.Counter()
        for _ in range(R):
            o = [rng.binomial(n, p) / n for p in (k0, k1, f0, f1)]
            bk0, bk1, bf0, bf1 = [rng.binomial(n, np.tile(p, (B, 1))) / n for p in o]
            F = (bf1 - bf0).mean(1); K = (bk0 - bk1).mean(1)
            lo, hi = np.percentile(F, [5, 95])
            out['YES' if (lo > b and np.percentile(K, 5) > 0) else 'NO' if hi < b else 'NS'] += 1
        print(f'n={n} {name:16s} true F {np.mean(f1 - f0):+.3f}: ' + ' '.join(f'{k} {out[k]/R:.2f}' for k in ('YES', 'NO', 'NS')))
