"""Nadia part 2: the 400-number gauge (W_oh row 0 shifted to 0), phantom-augmented oracles, second input seed.
Run: ../.venv/bin/python nadia_check2.py"""
import json
import numpy as np
from nadia_lib import Wi, Wh, Wo, fwd, sample, round_sig, gauge

res = {}
for seed in (432010, 432030):
    X = sample(seed)
    m = fwd(Wi, Wh, Wo, X)
    gi, gh, go = gauge(Wi, Wh, Wo)
    go0 = go - go[0:1]          # argmax-invariant: same vector added to every logit row
    r = {'gauge_shift_numbers': int((gh != 0).sum() + (go0 != 0).sum()),
         'gauge_shift_exact': float((fwd(gi, gh, go0, X) == m).mean()),
         'all400@3d': float((fwd(gi, round_sig(gh, 3), round_sig(go0, 3), X) == m).mean()),
         'all416@3d': float((fwd(gi, round_sig(gh, 3), round_sig(go, 3), X) == m).mean())}
    Z = np.concatenate([np.zeros((len(X), 1)), X], 1)
    o = np.argsort(-Z, axis=1)
    pos = np.where(o == 0, 0, o - 1)          # phantom decodes to position 0
    hit = lambda ks: np.any([pos[:, k] == m for k in ks], axis=0)
    r['phantom_oracle_S'] = float(hit([1]).mean())
    r['phantom_oracle_ST'] = float(hit([1, 2]).mean())
    r['phantom_oracle_MST'] = float(hit([0, 1, 2]).mean())
    if seed == 432030:
        for f, d in [('prune_ft_long_220_144_q3f.npz', None), ('ladder_pruned_0.99.npz', 4), ('mixprec_0.989.npz', None)]:
            z = np.load('../' + f)
            a, b, c = [z[k] for k in (('W_hi', 'W_hh', 'W_oh') if 'W_hi' in z else ('Wi', 'Wh', 'Wo'))]
            if d: b, c = round_sig(b, d), round_sig(c, d)
            r[f] = float((fwd(a, b, c, X) == m).mean())
    res[seed] = r
print(json.dumps(res, indent=1))
json.dump(res, open('nadia_check2.json', 'w'), indent=1)
