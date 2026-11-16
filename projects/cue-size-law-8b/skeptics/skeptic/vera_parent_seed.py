"""Vera: seed-level slope spread at 0.6B from the parent's raw curves, and power of the 8B status rule.

Own code. Reads parent run JSONs via git show (branch swarm/theory-1110-cue-size-learning).
"""
import json, subprocess, numpy as np
from scipy import stats

BR = "origin/swarm/theory-1110-cue-size-learning"
P = "projects/cue-size-law/data/runs/"
REPO = "/Users/esben/work/theory-1116-cue-size-law-8b/kran-research"


def load(name):
    out = subprocess.run(["git", "show", f"{BR}:{P}{name}.json"], cwd=REPO, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def frozen_grid(t):
    return t in (8, 16, 24, 32, 48, 64) or (t > 64 and t % 64 == 0)


def T_rule(curve, thr=0.10, grid="frozen"):
    pts = sorted({c["step"]: c for c in curve}.items())
    if grid == "frozen":
        pts = [(s, c) for s, c in pts if s == 0 or frozen_grid(s)]
    for i in range(len(pts) - 1):
        if pts[i][1]["D"] >= thr and pts[i + 1][1]["D"] >= thr:
            return pts[i][0]
    return None


def T_interp(curve, thr=0.10):
    """first upward crossing of D=thr, linearly interpolated, confirmed by the next eval >= thr (all evals)."""
    pts = sorted({c["step"]: c for c in curve}.items())
    for i in range(1, len(pts) - 1):
        if pts[i][1]["D"] >= thr and pts[i + 1][1]["D"] >= thr:
            s0, d0 = pts[i - 1][0], pts[i - 1][1]["D"]
            s1, d1 = pts[i][0], pts[i][1]["D"]
            return s0 + (thr - d0) / (d1 - d0) * (s1 - s0) if d1 != d0 else s1
    return None


alphas = [0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625]
rows = []
for k, a in enumerate(alphas):
    for s in (0, 1, 2):
        pre = "p0" if k == 0 else "sw"
        for name in (f"rd_{pre}_a{a}_s{s}", f"{pre}_a{a}_s{s}"):
            try:
                d = load(name)
            except subprocess.CalledProcessError:
                continue
            curve = d["dev_curve"] + d.get("dense_curve", [])
            rows.append(dict(name=name, k=k, seed=s, x=d["cue_size"].get("x", None), alpha=a,
                             Tf=T_rule(d["dev_curve"]), Tall=T_rule(curve, grid="all"), Ti=T_interp(curve),
                             has_dense="dense_curve" in d))
            break
for r in rows:
    print(r)

# per-seed slopes on log alpha (x is proportional to alpha up to <1%)
def slope(xs, ys):
    return np.polyfit(np.log(xs), np.log(ys), 1)[0]

res = {}
for key in ("Tf", "Tall", "Ti"):
    per = {}
    for s in (0, 1):
        rr = [r for r in rows if r["seed"] == s and r[key]]
        per[s] = slope([r["alpha"] for r in rr], [r[key] for r in rr])
    res[key] = per
    print(key, "per-seed slopes", {s: round(v, 4) for s, v in per.items()}, "diff", round(per[0] - per[1], 4))

# residual SD of log T around the pooled fit, per key
for key in ("Tf", "Tall", "Ti"):
    rr = [r for r in rows if r[key]]
    b, a0 = np.polyfit(np.log([r["alpha"] for r in rr]), np.log([r[key] for r in rr]), 1)
    resid = np.log([r[key] for r in rr]) - (a0 + b * np.log([r["alpha"] for r in rr]))
    print(key, "pooled slope", round(b, 4), "resid SD", round(resid.std(ddof=2), 4))
json.dump(dict(rows=rows, per_seed=res), open("vera_parent_seed.json", "w"), indent=1)
