"""Recompute the log-prob reading cells from Bruno's saved vectors <model>_A_reading.npz with own numpy + bootstrap."""
import json, numpy as np
from vera_stats_rms import boot_med
D = '../../kran-research/projects/frozen-norm-gradients/data/runs/'
out = {}
for model in ['qwen3', 'smol', 'gpt2']:
    z = np.load(f'{D}{model}_A_reading.npz'); o = {}
    Ls = sorted({int(k[1:].split('_')[0]) for k in z.files})
    for L in Ls:
        for tg in ['centred', 'logprob']:
            f, r = z[f'L{L}_{tg}_frz'].astype(np.float64), z[f'L{L}_{tg}_real'].astype(np.float64)
            c = (f * r).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(r, axis=1)
            m, lo, hi = boot_med(c); o[f'{L}_{tg}'] = [round(m, 4), round(lo, 4), round(hi, 4), len(c)]
    out[model] = o; print(model, o)
json.dump(out, open('vera_fromraw_reading.json', 'w'), indent=1)
