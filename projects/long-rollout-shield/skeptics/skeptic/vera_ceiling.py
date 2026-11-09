"""Vera: check Kasper's ceiling (#17569) on the parent's raw npz and price the PREREG v1 bars against it.
gap = E[(c_off - c_on) 1{texts differ}] <= f x range for the CRN coupling, and <= TV x range for the best coupling."""
import numpy as np, json
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
base = '/Users/esben/work/oracle-1108-multi-token/kran-research/projects/multi-token-recompute/data/reads/raw/'
out = {}
r = np.load(base + 'planted_s0.npz'); te = r['split'] == 'test'
g = (r['c_n'][te] - r['c_l'][te]).astype(float); fl = (r['toks_n'][te] != r['toks_l'][te]).any(1)
out['parent_planted'] = dict(gap=g.mean() * 100, f=fl.mean(), mean_gap_on_differing=g[fl].mean() * 100,
                             forced=float((r['c_notX'][te] - r['c_X'][te]).mean() * 100), D_end=0.4141,
                             ratio_gap_over_D=g.mean() / 0.4141 * 100)
rng_pts = out['parent_planted']['forced']
ceil = 0.104 * 42.1   # Kasper's range at start cheat 0.80
# planted at D 0.104, perfectly learned: rows differ ~D of the time, |g| ~ range there
f_p = 0.104; sd_row = np.sqrt(f_p * (0.429 ** 2) - (f_p * 0.429) ** 2)
se_p = sd_row / np.sqrt(1600) * 100
out['planted_matched'] = dict(ceiling=ceil, row_sd=sd_row, se=se_p,
                              P_point_ge5_if_at_ceiling=1 - Phi((5 - ceil) / se_p),
                              P_cilo_ge3_if_at_ceiling=1 - Phi((3 + 1.96 * se_p - ceil) / se_p))
se_r = 0.195 / np.sqrt(1600) * 100
for frac in (1.0, 0.9, 0.75, 0.5):
    t = ceil * frac
    p1 = 1 - Phi((3 + 1.96 * se_r - t) / se_r)
    out[f'real_YES_both_at_{frac}_of_ceiling'] = dict(true=t, P_one=p1, P_both=p1 ** 2)
print(json.dumps(out, indent=1, default=float)); json.dump(out, open('vera_ceiling.json', 'w'), indent=1, default=float)
