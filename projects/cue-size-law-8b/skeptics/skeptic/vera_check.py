"""Vera (skeptic) own checker for the cue-size law. Reads ONLY raw run JSONs (dev_curve [+ dense_curve]).

Recomputes from raw curves, independent of the build's analyze code:
  T      = first eval t with D(t) >= 0.10 and D(next eval) >= 0.10, over ALL saved evals (dense T), D = gap - gap(0)
  slope  = OLS of log T on log x per seed; seed-level 95% t-interval over seeds (df n-1); pooled OLS + run bootstrap
  null   = size-label permutation within seed (slope of shuffled labels), and the shuffled-coin control's max D
Usage: python3 vera_check.py RUNS_DIR [--prefix sw_] [--parent]
"""
import json, sys, glob, os, re, argparse
import numpy as np
from scipy import stats, optimize

ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("--glob", default="*.json")
ap.add_argument("--thr", type=float, default=0.10)
ap.add_argument("--xkey", default="x")
ap.add_argument("--out", default=None)
a = ap.parse_args()


def evals(d):
    c = {e["step"]: e for e in d.get("dev_curve", [])}
    for e in d.get("dense_curve", []) or []:
        c.setdefault(e["step"], e)
    pts = sorted(c.items())
    g0 = pts[0][1]["gap"] if pts and pts[0][0] == 0 else d.get("gap0")
    return [(s, e["gap"] - g0) for s, e in pts], g0  # D recomputed from gap, not the saved D


def T_of(pts, thr):
    for i in range(len(pts) - 1):
        if pts[i][1] >= thr and pts[i + 1][1] >= thr:
            return pts[i][0]
    return None


def T_interp(pts, thr):
    for i in range(1, len(pts) - 1):
        if pts[i][1] >= thr and pts[i + 1][1] >= thr:
            (s0, d0), (s1, d1) = pts[i - 1], pts[i]
            return s0 + (thr - d0) / (d1 - d0) * (s1 - s0) if d1 > d0 else s1
    return None


rows = []
for f in sorted(glob.glob(os.path.join(a.runs, a.glob))):
    d = json.load(open(f))
    args = d.get("args", {})
    pts, g0 = evals(d)
    if not pts:
        continue
    cs = d.get("cue_size", {})
    x = cs.get(a.xkey, cs.get("x"))
    rows.append(dict(run=os.path.basename(f)[:-5], seed=args.get("seed"), alpha=args.get("cue_alpha"), x=x,
                     arm=args.get("arm"), steps=pts[-1][0], T=T_of(pts, a.thr), Ti=T_interp(pts, a.thr),
                     maxD=max(p[1] for p in pts), finalD=pts[-1][1], gap0=g0, T_saved=d.get("T"),
                     n_evals=len(pts)))

for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def ols(lx, ly):
    b, c = np.polyfit(lx, ly, 1)
    return b, c


def fit(rows, key="T", xkey="alpha"):
    sw = [r for r in rows if r[key] and r[xkey]]
    seeds = sorted({r["seed"] for r in sw})
    per = {}
    for s in seeds:
        rr = [r for r in sw if r["seed"] == s]
        if len({r[xkey] for r in rr}) >= 3:
            per[s] = ols(np.log([r[xkey] for r in rr]), np.log([r[key] for r in rr]))[0]
    v = np.array(list(per.values()))
    n = len(v)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    lx = np.log([r[xkey] for r in sw]); ly = np.log([r[key] for r in sw])
    b, c = ols(lx, ly)
    # run bootstrap within size
    rng = np.random.default_rng(0)
    by = {}
    for r in sw:
        by.setdefault(r[xkey], []).append(r)
    bs = []
    for _ in range(2000):
        rr = [g[i] for g in by.values() for i in rng.integers(0, len(g), len(g))]
        bs.append(ols(np.log([r[xkey] for r in rr]), np.log([r[key] for r in rr]))[0])
    # size-label permutation null within seed
    perm = []
    for _ in range(2000):
        allb = []
        for s in per:
            rr = [r for r in sw if r["seed"] == s]
            xs = rng.permutation([r[xkey] for r in rr])
            allb.append(ols(np.log(xs), np.log([r[key] for r in rr]))[0])
        perm.append(np.mean(allb))
    perm = np.array(perm)
    return dict(key=key, xkey=xkey, n_runs=len(sw), per_seed={int(s): float(x) for s, x in per.items()},
                seed_mean=float(v.mean()), seed_ci=[float(v.mean() - h), float(v.mean() + h)],
                pooled=float(b), intercept=float(c), boot_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                perm_null_95=[float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))],
                perm_p=float(np.mean(np.abs(perm) >= abs(v.mean()))))


res = {}
for key in ("T", "Ti"):
    for xkey in ("alpha", "x"):
        try:
            res[f"{key}_{xkey}"] = fit(rows, key, xkey)
            print(json.dumps(res[f"{key}_{xkey}"]))
        except Exception as e:
            print(key, xkey, "fit failed", e)
if a.out:
    json.dump(dict(rows=rows, fits=res), open(a.out, "w"), indent=1)
