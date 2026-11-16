"""Vera robustness of the per-seed slope (not frozen; sensitivity only), own code from raw run JSONs.
(a) drop-one size, (b) top 3 vs bottom 3 sizes, (c) start-up offset: ln(T - c), c = 8, 16,
(d) quadratic curvature in ln share per model, (e) 8B minus matched 0.6B slope on each subset (Welch)."""
import json, glob, os, numpy as np
from scipy import stats
P = "/Users/esben/work/theory-1116-cue-size-law-8b/kran-research/projects/cue-size-law-8b/data/"


def T_of(d):
    c = {e["step"]: e for e in d["dev_curve"]}
    for e in d.get("dense_curve", []) or []:
        c.setdefault(e["step"], e)
    s = sorted(c); g0 = c[s[0]]["gap"]
    for i in range(len(s) - 1):
        if c[s[i]]["gap"] - g0 >= 0.1 and c[s[i + 1]]["gap"] - g0 >= 0.1:
            return s[i]


def load(sub, pre):
    out = {}
    for f in glob.glob(P + sub + "/" + pre + "_k*_s*.json"):
        d = json.load(open(f))
        if d["args"].get("arm") != "cue":
            continue
        out[(d["args"]["seed"], d["k"])] = (d["share"], T_of(d))
    return out


def slopes(runs, ks, c=0):
    res = []
    for s in sorted({s for s, _ in runs}):
        pts = [runs[(s, k)] for k in ks if (s, k) in runs and runs[(s, k)][1]]
        if len(pts) >= 3:
            res.append(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] - c for p in pts]), 1)[0])
    return np.array(res)


def tci(v):
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():+.3f} [{v.mean()-h:+.3f}, {v.mean()+h:+.3f}]"


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean(); h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    return f"{d:+.3f} [{d-h:+.3f}, {d+h:+.3f}]"


m8, m6 = load("runs", "m8"), load("runs_06b", "m06")
cases = [("all 6", range(6), 0)] + [(f"drop k{j}", [k for k in range(6) if k != j], 0) for j in range(6)] + \
        [("top k0-2", range(3), 0), ("bottom k3-5", range(3, 6), 0), ("offset 8", range(6), 8), ("offset 16", range(6), 16)]
out = {}
for name, ks, c in cases:
    a, b = slopes(m8, list(ks), c), slopes(m6, list(ks), c)
    inside = lambda v: (lambda m, h: (m - h >= -1.4) and (m + h <= -0.7))(v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    print(f"{name:12s} 8B {tci(a)} in-band {inside(a)} | 0.6B {tci(b)} | diff {welch(a, b)}")
    out[name] = dict(s8=a.tolist(), s06=b.tolist())
for lab, runs in (("8B", m8), ("0.6B", m6)):
    x = np.log([v[0] for v in runs.values()]); y = np.log([v[1] for v in runs.values()])
    X = np.c_[np.ones_like(x), x, x ** 2]
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    sig2 = res[0] / (len(y) - 3); cov = sig2 * np.linalg.inv(X.T @ X)
    print(f"{lab} quadratic ln T = a + b ln s + c (ln s)^2: c = {beta[2]:+.3f} (t {beta[2]/np.sqrt(cov[2,2]):.2f}); "
          f"local slope at top {beta[1]+2*beta[2]*x.max():+.3f}, at bottom {beta[1]+2*beta[2]*x.min():+.3f}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_robust.json"), "w"), indent=1)
