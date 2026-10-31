"""Vera (own code): dose number p per unit and pin from my clean pass (vera_clean.npz) and my r_hat (vera_r13.npy).
PREREG v1: p(unit, m) = mean over TEST prompts and prompt tokens t >= 1 of -(delta_t . r_hat)/||h_t||.
Two readings of 'mean over prompts and tokens': per-prompt mean then mean over prompts (pp), or one pooled token
mean (pool). Pure dose d = same mean of ||delta_t||/||h_t||. Only the gate and the natural neurons need their own
activations; any matched-push unit v (unit vector) has p = -(v . r_hat) * mean(n_t/||h_t||), n_t = |m - a_t| ||w||.
"""
import json, numpy as np
import vera_fwd as F

Z = np.load('vera_clean.npz', allow_pickle=True)
acts, hn = Z['acts'], Z['hn']
NEUR = list(Z['neur'])
rh = np.load('vera_r13.npy')
Wd = F.T('model.layers.13.mlp.down_proj.weight').numpy()
M5 = [6, 12, 18, 24, 30]


def tokmean(f, n=None):
    """f(prompt_index) -> per-token array over t>=1; returns (pp, pool)."""
    per = [f(b) for b in range(len(acts) if n is None else n)]
    return float(np.mean([x.mean() for x in per])), float(np.concatenate(per).mean())


out = {}
for k, j in enumerate(NEUR):
    w = Wd[:, j]
    wr, wn = float(w @ rh), float(np.linalg.norm(w))
    for m in M5:
        p = tokmean(lambda b: -(m - acts[b][1:, k]) * wr / hn[b][1:])
        d = tokmean(lambda b: np.abs(m - acts[b][1:, k]) * wn / hn[b][1:])
        p100 = tokmean(lambda b: -(m - acts[b][1:, k]) * wr / hn[b][1:], 100)
        plast = float(np.mean([-(m - acts[b][-1, k]) * wr / hn[b][-1] for b in range(len(acts))]))
        out[f'{j}:{m}'] = dict(cos=wr / wn, p_pp=p[0], p_pool=p[1], d_pp=d[0], d_pool=d[1], p_test100_pp=p100[0], p_last=plast)
g = {m: out[f'3270:{m}'] for m in M5}
for m in M5:
    print(f"gate m={m}: p_pp {g[m]['p_pp']:.4f} p_pool {g[m]['p_pool']:.4f} d_pp {g[m]['d_pp']:.4f} p_last {g[m]['p_last']:.4f}")
print('cos gate', g[30]['cos'])
for j in NEUR[1:]:
    print(j, 'cos %.3f' % out[f'{j}:30']['cos'], 'p_pp@30 %.4f' % out[f'{j}:30']['p_pp'], 'd_pp@30 %.4f' % out[f'{j}:30']['d_pp'])
json.dump(out, open('vera_p.json', 'w'), indent=1)
