"""Nadia (skeptic3) independent rebuild. Own forward from raw m16_10.pth; no core.py / check_agreement.py.
Seeds 4320xx only (never the 432432 holdout). Run: ../.venv/bin/python nadia_check.py
"""
import json
import numpy as np
import torch
from nadia_lib import sd, Wi, Wh, Wo, fwd, sample, round_sig, gauge

res = {}
# 0. forward check against torch nn.RNN (not alg_zoo)
rnn = torch.nn.RNN(1, 16, nonlinearity='relu', bias=False, batch_first=True).double()
lin = torch.nn.Linear(16, 10, bias=False).double()
rnn.weight_ih_l0.data = sd['rnn.weight_ih_l0'].double(); rnn.weight_hh_l0.data = sd['rnn.weight_hh_l0'].double()
lin.weight.data = sd['linear.weight'].double()
Xs = sample(432001, 50_000)
with torch.no_grad():
    o, hn = rnn(torch.from_numpy(Xs)[..., None])
    tp = lin(o[:, -1]).argmax(1).numpy()
res['torch_vs_numpy_agree'] = float((tp == fwd(Wi, Wh, Wo, Xs)).mean())

X = sample(432010)
m = fwd(Wi, Wh, Wo, X)
order = np.argsort(-X, axis=1)            # positions of max, 2nd, 3rd, ...
truth = order[:, 1]
res['model_acc'] = float((m == truth).mean())
err = m != truth
res['err_pick_3rd'] = float((m[err] == order[err, 2]).mean())
res['err_pick_max'] = float((m[err] == order[err, 0]).mean())
# oracle: perfect chooser within {2nd,3rd} and {max,2nd,3rd}
res['oracle_23'] = float(((m == order[:, 1]) | (m == order[:, 2])).mean())
res['oracle_123'] = float(((m == order[:, 0]) | (m == order[:, 1]) | (m == order[:, 2])).mean())

# C1 from its RESULTS.md text: 2nd largest of (0, x_0..x_9); the leading 0 decodes to position 0.
Z = np.concatenate([np.zeros((len(X), 1)), X], 1)
c1 = np.argsort(-Z, axis=1, kind='stable')[:, 1]
c1 = np.where(c1 == 0, 0, c1 - 1)
res['C1'] = float((c1 == m).mean())

# 0.1% multiplicative noise floor, 20 draws, two input seeds
for xs in (432010, 432020):
    Xn = X if xs == 432010 else sample(xs)
    mn = m if xs == 432010 else fwd(Wi, Wh, Wo, Xn)
    g = np.random.default_rng(xs + 7)
    ag = []
    for _ in range(20):
        e = lambda W: W * (1 + 1e-3 * g.standard_normal(W.shape))
        ag.append(float((fwd(e(Wi), e(Wh), e(Wo), Xn) == mn).mean()))
    res[f'floor_seed{xs}'] = {'mean': float(np.mean(ag)), 'sd': float(np.std(ag, ddof=1)), 'min': min(ag), 'max': max(ag)}

# all weights, +-1 gauge, 3 significant digits (zero fitted)
gi, gh, go = gauge(Wi, Wh, Wo)
r3h, r3o = round_sig(gh, 3), round_sig(go, 3)
res['all@3d_numbers'] = int((r3h != 0).sum() + (r3o != 0).sum())
res['all@3d'] = float((fwd(gi, r3h, r3o, X) == m).mean())
res['all@2d'] = float((fwd(gi, round_sig(gh, 2), round_sig(go, 2), X) == m).mean())
# null: stochastic rounding at 3 digits (is nearest-rounding lucky?)
g = np.random.default_rng(99)
def stoch(W, d):
    W = np.asarray(W, float); out = np.zeros_like(W); nz = W != 0
    q = 10 ** (np.floor(np.log10(np.abs(W[nz]))) - d + 1)
    lo = np.floor(W[nz] / q); p = W[nz] / q - lo
    out[nz] = (lo + (g.random(p.shape) < p)) * q
    return out
res['all@3d_stochastic_5draws'] = [float((fwd(gi, stoch(gh, 3), stoch(go, 3), X) == m).mean()) for _ in range(5)]

# candidate files, own forward
def cand(f, d=None):
    z = np.load('../' + f)
    a, b, c = [z[k] for k in (('W_hi', 'W_hh', 'W_oh') if 'W_hi' in z else ('Wi', 'Wh', 'Wo'))]
    if d: b, c = round_sig(b, d), round_sig(c, d)
    sig = lambda W: max(len(f'{v:.6e}'.split('e')[0].rstrip('0').replace('-', '').replace('.', '')) for v in W[W != 0])
    return {'numbers': int((b != 0).sum() + (c != 0).sum()), 'W_hi_vals': sorted(set(np.round(a, 6).tolist())),
            'max_sig_digits': sig(np.concatenate([b.ravel(), c.ravel()])),
            'agree_s432010': float((fwd(a, b, c, X) == m).mean())}
res['F3_364@3d'] = cand('prune_ft_long_220_144_q3f.npz')
res['F1_340@3d'] = cand('prune_ft_long_200_140_q3f.npz')
res['row3_374@4d'] = cand('ladder_pruned_0.99.npz', 4)
res['row0_mixprec'] = cand('mixprec_0.989.npz')
print(json.dumps(res, indent=1))
json.dump(res, open('nadia_check.json', 'w'), indent=1)
