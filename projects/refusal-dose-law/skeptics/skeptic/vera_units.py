"""Vera: check build's unit vectors (data/phase0/units.pt) against the PREREG definitions, with MY r_hat and MY clean
pass, and recompute every cell's p and pure dose d (gate-profile units: delta_t = (m - a_t^gate) v).
Compares to build's p_table.json cell means.
"""
import json, numpy as np, torch
import vera_fwd as F

D = '/Users/esben/work/theory-1031-refusal-dose-law/swarm-theory-1031-refusal-dose-law/data/'
U = torch.load(D + 'phase0/units.pt', weights_only=False)
P = json.load(open(D + 'phase0/p_table.json'))
rh = np.load('vera_r13.npy')
Z = np.load('vera_clean.npz', allow_pickle=True)
acts, hn = Z['acts'], Z['hn']
w = F.T('model.layers.13.mlp.down_proj.weight')[:, 3270].numpy()
wn = np.linalg.norm(w)
cg = float(w @ rh / wn)
# per-prompt gate profile means over t >= 1: mean (m - a_t)/||h_t||
prof = {m: np.array([((m - a[1:, 0]) / h[1:]).mean() for a, h in zip(acts, hn)]) for m in [6, 12, 18, 24, 30]}
rows, worst = [], 0
for name, (kind, v) in U.items():
    if not hasattr(v, "float"):
        continue
    v = v.float().numpy()
    vr = float(v @ rh)
    info = dict(norm_over_w=float(np.linalg.norm(v) / wn), cos_r=vr / np.linalg.norm(v))
    if kind == 'gate':
        for m in [6, 12, 18, 24, 30]:
            key = f'{name}@{m}'
            if key not in P:
                continue
            n = len(P[key]['p'])
            p_mine = float((-vr * prof[m][:n]).mean())
            d_mine = float((np.linalg.norm(v) * np.abs(prof[m][:n])).mean())
            p_b, d_b = float(np.mean(P[key]['p'])), float(np.mean(P[key]['d']))
            rel = abs(p_mine - p_b) / max(abs(p_b), 1e-3)
            worst = max(worst, rel if abs(p_b) > 1e-3 else 0)
            rows.append((key, round(p_mine, 5), round(p_b, 5), round(d_mine, 4), round(d_b, 4)))
    info['kind'] = kind
    print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in info.items()})
print('gate cos', cg)
print('worst relative p diff (|p|>1e-3):', worst)
for r in rows:
    if r[0].endswith('@30') or r[0].startswith(('G', 'TWIN', 'D_')):
        print(r)
json.dump(rows, open('vera_units.json', 'w'))
