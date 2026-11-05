"""Recompute status cells from Bruno's saved VECTORS (npz) with Vera's own numpy: cos(frz, real), tick cos(frz, pred),
bootstrap CI (own boot_med), selection check against my own selection rule (A: my rows). Usage: vera_fromraw.py MODEL SET..."""
import sys, json, numpy as np
from vera_stats_rms import boot_med
D = '../../kran-research/projects/frozen-norm-gradients/data/runs/'
model = sys.argv[1]; out = {}
for S in sys.argv[2:]:
    a = np.load(f'{D}{model}_{S}_s1.npz'); b = np.load(f'{D}{model}_{S}_s2.npz')
    keys1, keys2 = sorted(a.files), sorted(b.files)
    o = {'keys_s1': keys1, 'keys_s2': keys2, 'layers': {}}
    Ls = sorted({int(k[1:].split('_')[0]) for k in keys2 if k.startswith('L') and k.endswith('_real')})
    v = []
    for L in Ls:
        f, p, r = a[f'L{L}_frz'].astype(np.float64), a[f'L{L}_pred'].astype(np.float64), b[f'L{L}_real'].astype(np.float64)
        cs = lambda x, y: (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
        c = cs(f, r); cp = cs(f, p)
        m, lo, hi = boot_med(c); verdict = 'YES' if hi < 0.9 else ('NO' if lo >= 0.9 else 'unsettled')
        o['layers'][L] = {'n': len(c), 'median_c': round(m, 4), 'ci': [round(lo, 4), round(hi, 4)], 'verdict': verdict,
                          'median_cpred': round(float(np.median(cp)), 4), 'median_cos_pred_real': round(float(np.median(cs(p, r))), 4)}
        if L != max(Ls): v.append(verdict)
    o['status'] = 'CONFIRMED' if v.count('YES') >= 2 else ('REJECTED' if v.count('NO') >= 2 else 'UNCONFIRMED')
    out[S] = o; print(S, json.dumps(o['layers']), o['status'])
json.dump(out, open(f'vera_fromraw_{model}.json', 'w'), indent=1)
