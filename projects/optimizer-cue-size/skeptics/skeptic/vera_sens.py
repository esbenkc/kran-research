"""Vera: how fragile is the frozen status? (headline 24 runs, own T from the raw curves)

S1 switch level 0.05..0.15; S2 a fixed start-up offset c subtracted from every T (all optimizers);
S3 drop one cue size; S4 curvature (quadratic in ln x) per optimizer; S5 bootstrap P(d_SGD <= -0.6), P(|d_sign| <= 0.3);
S6 local slopes top-2 vs bottom-2 sizes. Status in S1-S3 uses the frozen rule with the pooled t interval.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_check as vc  # noqa: E402

OPTS = ["adamw", "sgd", "sign"]


def rows(thr=0.10, off=0.0, drop=None, sgd_lr=0.01):
    out = []
    for f in sorted(glob.glob(f"{vc.P}/data/runs/*.json")):
        r = json.load(open(f))
        a = r["args"]
        if a.get("arm") != "cue" or a.get("freeze_a") or a.get("group") != 8 or not a.get("no_cast"):
            continue
        if not any(abs(a["cue_alpha"] - s) / s < 1e-6 for s in vc.SIZES) or a["seed"] not in (0, 1):
            continue
        if a["opt"] == "sgd" and abs(a["lr"] - sgd_lr) > 1e-12:
            continue
        if a["opt"] in ("adamw", "sign") and abs(a["lr"] - 1e-4) > 1e-12:
            continue
        if drop is not None and abs(a["cue_alpha"] - drop) < 1e-12:
            continue
        T, c = vc.my_T(vc.merged_curve(r), thr)
        assert not c
        out.append((a["opt"], float(np.log(r["cue_size"]["x"])), float(np.log(T - off))))
    return out


def fit(rr):
    b, cov, df, sd = vc.ols(rr, OPTS)
    q = stats.t.ppf(0.975, df)
    con = {}
    for o in ("sgd", "sign"):
        d, se = vc.contrast(b, cov, OPTS, o, "adamw")
        con[o] = [d, d - q * se, d + q * se]
    st = vc.status(con["sgd"][1:], con["sgd"][0], con["sign"][1:], con["sign"][0])
    return {"slopes": {o: b[2 * i + 1] for i, o in enumerate(OPTS)}, "d": con, "status": st, "n": len(rr)}


res = {"S1": {}, "S2": {}, "S3": {}}
for thr in (0.05, 0.075, 0.10, 0.125, 0.15):
    res["S1"][thr] = fit(rows(thr=thr))
for off in (0, 4, 8, 12, 16):
    res["S2"][off] = fit(rows(off=off))
for a in vc.SIZES:
    res["S3"][a] = fit(rows(drop=a))
for k in ("S1", "S2", "S3"):
    for key, v in res[k].items():
        print(f"{k} {key}: slopes " + " ".join(f"{o} {v['slopes'][o]:.3f}" for o in OPTS)
              + f" | d_SGD {v['d']['sgd'][0]:.3f} [{v['d']['sgd'][1]:.3f},{v['d']['sgd'][2]:.3f}]"
              + f" d_sign {v['d']['sign'][0]:.3f} [{v['d']['sign'][1]:.3f},{v['d']['sign'][2]:.3f}] -> {v['status']}")
rr = rows()
res["S4"], res["S6"] = {}, {}
for o in OPTS:
    lx = np.array([r[1] for r in rr if r[0] == o])
    ly = np.array([r[2] for r in rr if r[0] == o])
    X = np.c_[np.ones_like(lx), lx - lx.mean(), (lx - lx.mean()) ** 2]
    b, *_ = np.linalg.lstsq(X, ly, rcond=None)
    e = ly - X @ b
    df = len(ly) - 3
    cov = (e @ e / df) * np.linalg.inv(X.T @ X)
    t = b[2] / np.sqrt(cov[2, 2])
    res["S4"][o] = {"quad": b[2], "t": t, "p": 2 * stats.t.sf(abs(t), df)}
    top = lx > np.median(lx)
    res["S6"][o] = {"top2": np.polyfit(lx[top], ly[top], 1)[0], "bottom2": np.polyfit(lx[~top], ly[~top], 1)[0]}
    print(f"S4 {o}: quadratic coef {b[2]:.3f} t {t:.2f} p {res['S4'][o]['p']:.4f} | S6 local slope top-2 "
          f"{res['S6'][o]['top2']:.3f} bottom-2 {res['S6'][o]['bottom2']:.3f}")
rng = np.random.default_rng(1)
cells = {}
for i, r in enumerate(rr):
    cells.setdefault((r[0], round(r[1], 3)), []).append(i)
ds, dg = [], []
for _ in range(10000):
    idx = [rng.choice(v) for v in cells.values() for _ in v]
    s = [rr[i] for i in idx]
    b, *_ = np.linalg.lstsq(vc.design(s, OPTS), np.array([x[2] for x in s]), rcond=None)
    ds.append(b[3] - b[1])
    dg.append(b[5] - b[1])
res["S5"] = {"P_dsgd_le_-0.6": float(np.mean(np.array(ds) <= -0.6)), "P_absdsign_le_0.3": float(np.mean(np.abs(dg) <= 0.3))}
print("S5 bootstrap:", res["S5"])
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_sens.json"), "w"), indent=1,
          default=float)
