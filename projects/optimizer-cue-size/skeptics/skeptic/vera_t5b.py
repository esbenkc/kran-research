"""Vera: rebuild of the exploratory arms (T5b) from the raw run JSONs, own T rule (vera_check.my_T on the union curve).

E1 D4 time-matched AdamW (lr 1e-5): pooled 4-arm OLS (AdamW, SGD, sign, AdamW-tm), shared SD; contrasts and D4 bar.
E2 offset model check: AdamW T = c + k/lr at x_top from the lr ladder; ratio (T_default - c)/(T_tm - c) per size.
E3 controls: SGD lr/2 at alpha 1/64 (band [1.6, 2.5]); SGD group 16 (drift band [0.7, 1.4]); shuffled coin (no switch).
E4 SGD at lr 0.02, two-point slope x33 -> x16.6 vs lr 0.01; A-frozen same-lr ratio.
E5 alpha 1/512 cells: 5-size slopes per optimizer and SGD local slope 4.16 -> 2.08.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_check as vc  # noqa: E402

R = {}
for f in sorted(glob.glob(f"{vc.P}/data/runs/*.json")):
    r = json.load(open(f))
    a = r["args"]
    T, c = vc.my_T(vc.merged_curve(r))
    td = r.get("T_dense")
    assert (c and td is None) or (not c and td == T), (f, T, td)
    R[os.path.basename(f)[:-5]] = dict(opt=a["opt"], lr=a["lr"], alpha=a["cue_alpha"], seed=a["seed"],
                                       fa=a.get("freeze_a"), G=a.get("group"), arm=a.get("arm"), nc=a.get("no_cast"),
                                       x=r["cue_size"]["x"], T=T, cens=c,
                                       maxD=max(cc["gap"] for cc in vc.merged_curve(r)) - vc.merged_curve(r)[0]["gap"])
print(f"{len(R)} runs, own T = T_dense in all (asserted)")
out = {}


def label(v):
    if v["arm"] != "cue" or v["fa"] or v["G"] != 8 or not v["nc"] or v["seed"] not in (0, 1):
        return None
    if not any(abs(v["alpha"] - s) / s < 1e-6 for s in vc.SIZES):
        return None
    if v["opt"] == "adamw":
        return {1e-4: "adamw", 1e-5: "adamwtm"}.get(v["lr"])
    if v["opt"] == "sgd":
        return "sgd" if abs(v["lr"] - 0.01) < 1e-12 else None
    return "sign"


rows = [(label(v), np.log(v["x"]), np.log(v["T"])) for v in R.values() if label(v)]
opts = ["adamw", "sgd", "sign", "adamwtm"]
b, cov, df, sd = vc.ols(rows, opts)
q = stats.t.ppf(0.975, df)
out["E1"] = {"n": len(rows), "df": df, "sd": sd, "slopes": {}, "contrasts": {}}
for i, o in enumerate(opts):
    se = np.sqrt(cov[2 * i + 1, 2 * i + 1])
    out["E1"]["slopes"][o] = [b[2 * i + 1], b[2 * i + 1] - q * se, b[2 * i + 1] + q * se]
for a_, c_ in (("adamwtm", "sgd"), ("adamwtm", "adamw"), ("sgd", "adamw")):
    d, se = vc.contrast(b, cov, opts, a_, c_)
    out["E1"]["contrasts"][f"{a_}-{c_}"] = [d, d - q * se, d + q * se]
tm = out["E1"]["slopes"]["adamwtm"]
sgd_pt = out["E1"]["slopes"]["sgd"][0]
ad_pt = out["E1"]["slopes"]["adamw"][0]
out["E1"]["D4_bar"] = ("step normalisation" if not (tm[1] <= sgd_pt <= tm[2])
                       else ("time regime" if not (tm[1] <= ad_pt <= tm[2]) else "not decided"))
print("E1", json.dumps(out["E1"], default=lambda z: round(float(z), 3)))

lad = sorted((v["lr"], v["T"]) for n, v in R.items() if v["opt"] == "adamw" and v["alpha"] == 2 ** -5 and v["seed"] == 0
             and v["nc"] and v["arm"] == "cue" and (n.startswith("adamwtmpilot") or n == "adamw_a0.03125_s0"))
lr = np.array([z[0] for z in lad])
T = np.array([z[1] for z in lad], float)
(c0, k), *_ = np.linalg.lstsq(np.c_[np.ones_like(lr), 1 / lr], T, rcond=None)


def gm(o, lrv, a):
    v = [z["T"] for z in R.values() if z["opt"] == o and abs(z["lr"] - lrv) < 1e-12 and abs(z["alpha"] - a) < 1e-12
         and z["seed"] in (0, 1) and z["arm"] == "cue" and not z["fa"] and z["G"] == 8 and z["nc"]]
    return float(np.exp(np.mean(np.log(v))))


ratio = [(gm("adamw", 1e-4, a) - c0) / (gm("adamw", 1e-5, a) - c0) for a in vc.SIZES]
out["E2"] = {"ladder": lad, "c": c0, "k": k, "ratio_by_size": ratio}
print(f"E2 c {c0:.1f} k {k:.3e}; (T_default - c)/(T_tm - c) by size x33..x4.2: {np.round(ratio, 3).tolist()} (1/lr model: 0.1)")

t = {n: v["T"] for n, v in R.items()}
out["E3"] = {"lr_half": t["ctl_lrhalf_sgd_a0.015625_s0"] / t["sgd_a0.015625_s0"],
             "g16": t["ctl_g16_sgd_a0.015625_s0"] / t["sgd_a0.015625_s0"],
             "shuffled_switched": not R["ctl_shuf_sgd_a0.03125_s0"]["cens"],
             "shuffled_maxD": R["ctl_shuf_sgd_a0.03125_s0"]["maxD"]}
print("E3", {k_: round(float(v), 4) for k_, v in out["E3"].items()})


def two_pt(n1, n2):
    return -np.log(t[n2] / t[n1]) / np.log(R[n1]["x"] / R[n2]["x"])


out["E4"] = {"sgd_lr0.02_2pt": two_pt("sgdpilot_lr0.02", "vera_sgdlr0.02_a0.015625_s0"),
             "sgd_lr0.01_2pt_s0": two_pt("sgdpilot_lr0.01", "sgd_a0.015625_s0"),
             "afrozen_ratio": t["sgdfa_samelr_a0.03125_s0"] / t["sgdpilot_lr0.01"]}
print("E4", {k_: round(float(v), 3) for k_, v in out["E4"].items()})

out["E5"] = {}
for o, lrv in (("adamw", 1e-4), ("sign", 1e-4), ("sgd", 0.01)):
    pts = [(np.log(v["x"]), np.log(v["T"]), v["cens"]) for v in R.values() if v["opt"] == o and abs(v["lr"] - lrv) < 1e-12
           and v["seed"] in (0, 1) and v["arm"] == "cue" and not v["fa"] and v["G"] == 8 and v["nc"]
           and any(abs(v["alpha"] - s) / s < 1e-6 for s in vc.SIZES + [2 ** -9])]
    lx = np.array([p[0] for p in pts])
    ly = np.array([p[1] for p in pts])
    small = [p for p in pts if abs(np.exp(p[0]) - 2.0793) < 0.01]
    out["E5"][o] = {"n": len(pts), "slope_5size": float(np.polyfit(lx, ly, 1)[0]), "x2.06": [float(np.exp(p[1])) for p in small],
                    "censored_x2.06": [bool(p[2]) for p in small]}
    if small:
        out["E5"][o]["local_4.16_to_2.08"] = float(-np.log(np.exp(np.mean([p[1] for p in small])) / gm(o, lrv, 2 ** -8)) / np.log(2))
print("E5", json.dumps(out["E5"], default=lambda z: round(float(z), 3)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_t5b.json"), "w"), indent=1, default=float)
