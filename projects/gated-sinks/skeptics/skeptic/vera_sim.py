"""Vera: price the frozen law test under a null and under real effects (C5, Tove #17129).

Rule 'felix' (#17131): fit recovery = a + b*sink on the NON-gated models only.
  band = max(0.03, 2*sd_sae)
  CONFIRMED  : 95% CI of b excludes 0 with b < 0 AND >= 2 of 3 gated within band of the line
  REJECTED   : (i) 95% CI of b*sink_range inside +-0.05, OR (ii) >= 2 of 3 gated more than 2*band off
  UNCONFIRMED: else
Rule 'tove' (#17129): CONFIRMED = b CI < 0 and every gated in band; REJECTED = |b_hat*range| < band
  or any gated off band.
CI: OLS t-interval (n small; a bootstrap over 9 points is no better). Noise: each model's recovery
= truth + model-seed noise + mean of n_sae SAE-seed noises.
"""
import argparse, json
import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--ng_sink", default="0.45,0.45,0.45,0.25,0.25,0.05,0.05,0.03,0.03")
p.add_argument("--g_sink", default="0.05,0.05,0.05")
p.add_argument("--sd_model", type=float, default=0.02)
p.add_argument("--sd_sae", type=float, default=0.01)
p.add_argument("--n_sae", type=int, default=3)
p.add_argument("--slope", type=float, default=0.0, help="true recovery change per unit sink (null=0)")
p.add_argument("--gated_offset", type=float, default=0.0, help="gated off-line effect (gate sparsity)")
p.add_argument("--rule", default="prereg", choices=["prereg", "felix", "tove"])
p.add_argument("--eq", type=float, default=0.05, help="floor of the REJECTED (i) swing bar")
p.add_argument("--n", type=int, default=20000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
xs = np.array([float(v) for v in a.ng_sink.split(",")])
xg = np.array([float(v) for v in a.g_sink.split(",")])
band = max(0.03, 2 * a.sd_sae)
srange = np.ptp(np.r_[xs, xg])
sd_unit = np.sqrt(a.sd_model**2 + a.sd_sae**2 / a.n_sae)
tcrit = stats.t.ppf(0.975, xs.size - 2)
out = {"CONFIRMED": 0, "REJECTED": 0, "UNCONFIRMED": 0}
for _ in range(a.n):
    ys = 0.7 + a.slope * xs + rng.normal(0, sd_unit, xs.size)
    yg = 0.7 + a.slope * xg + a.gated_offset + rng.normal(0, sd_unit, xg.size)
    r = stats.linregress(xs, ys)
    lo, hi = r.slope - tcrit * r.stderr, r.slope + tcrit * r.stderr
    off = np.abs(yg - (r.intercept + r.slope * xg))
    if a.rule == "prereg":  # PREREG.md a65167e6 section 3, exact
        res_ = ys - (r.intercept + r.slope * xs)
        s_tot = np.sqrt((res_**2).sum() / (xs.size - 2))
        h = tcrit * s_tot * np.sqrt(1 / xs.size + (xg - xs.mean())**2 / ((xs - xs.mean())**2).sum())
        bandg = np.maximum(max(0.03, 2 * s_tot), h)
        inb = off <= bandg
        R = np.ptp(xs)
        conf = hi < 0 and inb.sum() >= 2
        rej = (max(abs(lo), abs(hi)) * R < a.eq) or (~inb).sum() >= 2 or lo > 0
    elif a.rule == "felix":
        conf = hi < 0 and (off <= band).sum() >= 2
        rej = (max(abs(lo), abs(hi)) * srange < 0.05) or (off > 2 * band).sum() >= 2
    else:
        conf = hi < 0 and np.all(off <= band)
        rej = abs(r.slope * srange) < band or not np.all(off <= band)
    out["REJECTED" if rej else "CONFIRMED" if conf else "UNCONFIRMED"] += 1  # PREREG d890fd89: REJECTED first
res = {k: round(v / a.n, 3) for k, v in out.items()}
res.update(band=band, sd_unit=round(float(sd_unit), 4), sink_range=round(float(srange), 3))
print(json.dumps(res))
