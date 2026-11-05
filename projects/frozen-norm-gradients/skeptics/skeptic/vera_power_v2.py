"""Pre-data power sim of PREREG_v2 s5 status rule (no held-out data used).
Per layer: D_t ~ Normal(Delta_L, sdD) with heavy tail (t5), rho_L true Spearman; 200 tokens; percentile
bootstrap 2,000 resamples (10,000 in the prereg; CI widths converge). YES-layer: CI(Delta) in [-.05,.05] and
CI-low(rho) >= .70; NO-layer: CI(Delta) outside the band or CI-high(rho) < .70.
Scenarios: Delta_L grid, and a prior centred on the around-real screen gaps (vera_first_screen.json)."""
import numpy as np, json
from scipy.stats import spearmanr, rankdata

rng = np.random.default_rng(7)
N, B = 200, 1000


def layer_verdict(delta, sdD, rho):
    # build c_meas and c_pred with Spearman ~ rho and D_t = c_pred - c_meas with median ~ delta
    z1 = rng.standard_normal(N); z2 = rng.standard_normal(N)
    c_meas = 0.75 + 0.15 * z1
    D = delta + sdD * rng.standard_t(5, N) / np.sqrt(5 / 3)
    c_pred = c_meas + D
    # enforce target rank correlation by mixing noise into c_pred ranks only via D; measure realised rho
    idx = rng.integers(0, N, (B, N))
    med = np.median(D[idx], 1)
    r = np.array([spearmanr(c_pred[i], c_meas[i])[0] for i in idx[:300]])
    lo, hi = np.percentile(med, [2.5, 97.5]); rlo, rhi = np.percentile(r, [2.5, 97.5])
    yes = (lo >= -0.05) and (hi <= 0.05) and (rlo >= 0.70)
    no = (lo > 0.05) or (hi < -0.05) or (rhi < 0.70)
    return 'Y' if yes else ('N' if no else 'U'), float(spearmanr(c_pred, c_meas)[0]), hi - lo


def status(deltas, sdD):
    v = [layer_verdict(d, sdD, None) for d in deltas]
    ny = sum(x[0] == 'Y' for x in v); nn = sum(x[0] == 'N' for x in v)
    return ('CONFIRMED' if ny >= 2 else 'REJECTED' if nn >= 2 else 'UNCONFIRMED'), v


out = {}
# 1) single-layer band map
for sdD in [0.05, 0.10, 0.15]:
    row = {}
    for d in [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09]:
        vs = [layer_verdict(d, sdD, None) for _ in range(40)]
        row[d] = {k: sum(x[0] == k for x in vs) / 40 for k in 'YNU'}
        row[d]['rho'] = round(float(np.mean([x[1] for x in vs])), 3)
        row[d]['ci_w'] = round(float(np.mean([x[2] for x in vs])), 3)
    out[f'band_sdD{sdD}'] = row
    print(sdD, row, flush=True)
# 2) prior: |Delta_L| ~ |N(m_L, 0.04)| with m = around-real screen gaps scaled by s ~ U(0.3, 1.2), shared s
cnt = {'CONFIRMED': 0, 'REJECTED': 0, 'UNCONFIRMED': 0}
for _ in range(150):
    s = rng.uniform(0.3, 1.2); sdD = rng.uniform(0.05, 0.15)
    m = np.array([0.11, 0.09, 0.05]) * s
    deltas = np.abs(m + 0.03 * rng.standard_normal(3))
    cnt[status(deltas, sdD)[0]] += 1
out['prior'] = {k: v / 150 for k, v in cnt.items()}
print('prior', out['prior'])
json.dump(out, open('vera_power_v2.json', 'w'), indent=1, default=str)
