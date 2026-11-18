"""Vera (skeptic), room theory-1118-cue-size-law-8b: own rebuild from raw run JSONs only.

Reads data/runs/*.json and data/runs_06b/*.json of projects/cue-size-law-8b (no project code imported).
T (dense) = first eval t on the dense grid (every 8 updates to 2,048, then every 64) with
D(t) >= 0.10 and D(next) >= 0.10, D = gap(t) - gap(0), gap = P(X|cue off) - P(X|cue on), recomputed from px_off/px_on.
Per-seed slope = OLS of ln T on ln share (measured share per run). Seed-level 95% = t-interval over seeds.
Usage: python vera_1118.py <project dir> [out.json]
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

P = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None


def load(sub):
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{P}/data/{sub}/*.json")}


def curve(r):
    pts = {}
    for c in r["dev_curve"] + r["dense_curve"]:
        pts[c["step"]] = c["px_off"] - c["px_on"]
    return pts


def dense_T(r):
    pts = curve(r)
    g0 = pts[0]
    grid = [s for s in sorted(pts) if (s <= 2048 and s % 8 == 0) or (s > 2048 and s % 64 == 0)]
    D = [pts[s] - g0 for s in grid]
    for i in range(len(grid) - 1):
        if D[i] >= 0.10 and D[i + 1] >= 0.10:
            return grid[i], grid
    return None, grid


def interp_T(r):  # S7: linear crossing of D = 0.10 before the dense T
    t, grid = dense_T(r)
    pts = curve(r)
    g0 = pts[0]
    i = grid.index(t)
    if i == 0:
        return float(t)
    a = grid[i - 1]
    Da, Db = pts[a] - g0, pts[t] - g0
    return a + (0.10 - Da) * (t - a) / (Db - Da)


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [m, m - h, m + h]


runs, r6 = load("runs"), load("runs_06b")
res = {"T_mismatch": []}
for n, r in {**runs, **r6}.items():
    t, grid = dense_T(r)
    if t != r.get("T_dense"):
        res["T_mismatch"].append([n, t, r.get("T_dense")])
    assert set(np.diff([s for s in grid if s <= 2048])) == {8}, n

g8 = lambda k, s: runs[f"{'m8' if s < 3 else 'l8'}_k{k}_s{s}"]  # mini-primary seeds 0-2, laptop seeds 3-4
g6 = lambda k, s: r6[f"m06_k{k}_s{s}"]


def per_seed(get, seeds, fT=lambda r: dense_T(r)[0], ks=range(6)):
    out = {}
    for s in seeds:
        sh = np.array([get(k, s)["share"] for k in ks])
        T = np.array([fT(get(k, s)) for k in ks], float)
        assert all(abs(get(k, s)["gap0"]) < 0.02 for k in ks)
        out[s] = dict(T=T.tolist(), slope=float(np.polyfit(np.log(sh), np.log(T), 1)[0]),
                      c2=float(np.polyfit(np.log(sh), np.log(T), 2)[0]),
                      big=float(np.polyfit(np.log(sh[:3]), np.log(T[:3]), 1)[0]),
                      small=float(np.polyfit(np.log(sh[3:]), np.log(T[3:]), 1)[0]))
    return out


for name, get in [("8B", g8), ("0.6B", g6)]:
    ps = per_seed(get, range(5))
    b = [v["slope"] for v in ps.values()]
    res[name] = dict(per_seed=ps, seed_level=tci(b), per_halving=[2 ** -x for x in tci(b)],
                     c2=tci([v["c2"] for v in ps.values()]), big_half=tci([v["big"] for v in ps.values()]),
                     small_half=tci([v["small"] for v in ps.values()]),
                     interp_T=tci([v["slope"] for v in per_seed(get, range(5), interp_T).values()]),
                     drop_k0=tci([v["slope"] for v in per_seed(get, range(5), ks=range(1, 6)).values()]))
res["8B"]["seeds0_2"] = tci([res["8B"]["per_seed"][s]["slope"] for s in range(3)])
res["8B"]["frozen_grid_T"] = tci([v["slope"] for v in per_seed(g8, range(5), lambda r: r["T"]).values()])

# held-out: predictive interval for one new seed from the 3 seeds the live post published
b3 = np.array([res["8B"]["per_seed"][s]["slope"] for s in range(3)])
h = stats.t.ppf(0.975, 2) * b3.std(ddof=1) * np.sqrt(1 + 1 / 3)
res["heldout"] = dict(pred_interval=[b3.mean() - h, b3.mean() + h],
                      seeds3_4=[res["8B"]["per_seed"][s]["slope"] for s in (3, 4)])

# S1 Welch
b8 = np.array([v["slope"] for v in res["8B"]["per_seed"].values()])
b6 = np.array([v["slope"] for v in res["0.6B"]["per_seed"].values()])
v1, v2 = b8.var(ddof=1) / 5, b6.var(ddof=1) / 5
df = (v1 + v2) ** 2 / (v1 ** 2 / 4 + v2 ** 2 / 4)
d = b8.mean() - b6.mean()
hw = stats.t.ppf(0.975, df) * np.sqrt(v1 + v2)
res["S1_welch"] = dict(diff=d, ci=[d - hw, d + hw], df=df)

# S2 calibrated constant (slope -0.903, intercept = median T at s_0)
Tm = np.array([res["8B"]["per_seed"][s]["T"] for s in range(5)])
sh0 = np.mean([[g8(k, s)["share"] for k in range(6)] for s in range(5)], axis=0)
med = np.median(Tm, axis=0)
pred = med[0] * (sh0 / sh0[0]) ** -0.903
res["S2"] = dict(median_T=med.tolist(), pred=pred.tolist(), ratio=(med / pred).tolist())

# controls
ctl = {}
for n in ["m8_c1_s0", "l8_c1_s0", "l8_c2_s0"]:
    pts = curve(runs[n])
    ctl[n] = dict(T=dense_T(runs[n])[0], maxD=max(pts[s] - pts[0] for s in pts), max_abs_gap=max(abs(v) for v in pts.values()),
                  last=max(pts), n_evals=len(pts))
A, B = curve(runs["m8_c1_s0"]), curve(runs["l8_c1_s0"])
ctl["c1_resumed_vs_laptop_max_diff"] = max(abs(A[t] - B[t]) for t in A)
ctl["mini_vs_laptop_seeds0_2_max_gap_diff"] = max(
    max(abs(curve(runs[f"m8_k{k}_s{s}"])[t] - curve(runs[f"l8_k{k}_s{s}"])[t]) for t in curve(runs[f"m8_k{k}_s{s}"]))
    for s in range(3) for k in range(6))
res["controls"] = ctl

# null: permute T across sizes within each seed (20,000 draws)
rng = np.random.default_rng(1118)
sh = {s: np.log([g8(k, s)["share"] for k in range(6)]) for s in range(5)}
lT = {s: np.log(res["8B"]["per_seed"][s]["T"]) for s in range(5)}
null = np.array([np.mean([np.polyfit(sh[s], rng.permutation(lT[s]), 1)[0] for s in range(5)]) for _ in range(20000)])
res["perm_null"] = dict(mean=float(null.mean()), q025_975=np.percentile(null, [2.5, 97.5]).tolist(),
                        p_le_obs=float((null <= b8.mean()).mean()), n=20000)

print(json.dumps(res, indent=1, default=float))
if OUT:
    json.dump(res, open(OUT, "w"), indent=1, default=float)
