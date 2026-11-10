"""S2 read-off (PREREG s6 S2, s9.7): primary fit on AdamW random-direction (u seed 1110, +u) sweep runs only,
OLS of log T on log x over uncensored runs; predicted T at x_real with the wider of the bootstrap and t intervals.
Felix (plan), independent of Bruno's analyze.py.  python3 s2_readoff.py <project dir>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

P = Path(sys.argv[1] if len(sys.argv) > 1 else "kran-research/projects/cue-size-law")
scan = json.load(open(P / "data/scan.json"))
xr = scan["x_real"]
rows = []
for f in sorted(glob.glob(str(P / "data/runs/*.json"))):
    d = json.load(open(f))
    a = d["args"]
    if a.get("arm") != "cue" or a.get("cue_dir") != "rand" or a.get("opt") != "adamw":
        continue
    if a.get("u_file") != "data/cue_u.npy" or a.get("cue_sign", 1) != 1:
        continue
    if not (d["name"].startswith("p0_") or d["name"].startswith("sw_")):  # frozen grid only (k=0..5)
        continue
    al = a["cue_alpha"]
    x = scan["cues"][repr(al) if repr(al) in scan["cues"] else str(al)]["x"]
    rows.append((d["name"], al, x, d["T"], d["censored"], abs(d["gap0"])))
for r in rows:
    print(r)
u = [(al, x, T) for _, al, x, T, c, g0 in rows if not c and g0 < 0.02]
lx = np.log([r[1] for r in u]); lt = np.log([r[2] for r in u]); al = np.array([r[0] for r in u])
n = len(u)
res = stats.linregress(lx, lt)
tq = stats.t.ppf(0.975, n - 2)
pred = res.intercept + res.slope * np.log(xr)
xm = lx.mean(); sxx = ((lx - xm) ** 2).sum(); s = np.sqrt(((lt - res.intercept - res.slope * lx) ** 2).sum() / (n - 2))
se_mean = s * np.sqrt(1 / n + (np.log(xr) - xm) ** 2 / sxx)
rng = np.random.default_rng(0)
groups = {a_: np.where(al == a_)[0] for a_ in np.unique(al)}
bs, bp = [], []
for _ in range(2000):
    idx = np.concatenate([rng.choice(g, len(g)) for g in groups.values()])
    r2 = stats.linregress(lx[idx], lt[idx])
    bs.append(r2.slope); bp.append(r2.intercept + r2.slope * np.log(xr))
out = {
    "n_runs": n, "alphas": sorted(set(al.tolist())), "slope": res.slope,
    "slope_t95": [res.slope - tq * res.stderr, res.slope + tq * res.stderr],
    "slope_boot95": list(np.percentile(bs, [2.5, 97.5])),
    "x_real": xr, "T_pred_real": float(np.exp(pred)),
    "T_pred_real_t95": [float(np.exp(pred - tq * se_mean)), float(np.exp(pred + tq * se_mean))],
    "T_pred_real_boot95": [float(np.exp(v)) for v in np.percentile(bp, [2.5, 97.5])],
    "S2_window_2x": [float(np.exp(pred) / 2), float(np.exp(pred) * 2)],
}
print(json.dumps(out, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
