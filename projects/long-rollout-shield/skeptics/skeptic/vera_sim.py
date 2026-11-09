"""Vera: price the frozen 1.5-pt 'no switch' rule at k=64 before data.
Noise from the parent's start_k64 raw npz (row SD of the CRN-paired gap, 200 test prompts x R=2, rows ~independent
within prompt). Simulates per-seed point estimate and 95% CI for R in {2,4,8}, true gaps 0..6 pts, and the
2-of-2-seed rules: point<1.5, CI-hi<1.5 ('NO'), CI-lo>=3 ('YES')."""
import numpy as np, json
r = np.load('/Users/esben/work/oracle-1108-multi-token/kran-research/projects/multi-token-recompute/data/reads/raw/start_k64.npz')
te = r['split'] == 'test'
g = (r['c_n'][te] - r['c_l'][te]).astype(float)
sd = g.std(); P = 200
rng = np.random.default_rng(0); out = {'row_sd': sd}
for R in (2, 4, 8):
    se = sd / np.sqrt(P * R) * 100
    for scale in (1.0, 0.5, 1.5):   # collapse halves noise, open text may raise it
        s = se * scale
        for true in (0, 1, 2, 3, 4.4, 6):
            est = true + s * rng.standard_normal((20000, 2))
            k = f'R{R}_sd{scale}_true{true}'
            out[k] = dict(se=round(s, 2), half=round(1.96 * s, 2),
                          P_point_lt1p5_both=float((est < 1.5).all(1).mean()),
                          P_cihi_lt1p5_both=float((est + 1.96 * s < 1.5).all(1).mean()),
                          P_cilo_ge3_any=float((est - 1.96 * s >= 3).any(1).mean()))
json.dump(out, open('vera_sim.json', 'w'), indent=1)
for k, v in out.items():
    if k != 'row_sd' and '_sd1.0_' in k: print(k, v)
print('row_sd', sd)
