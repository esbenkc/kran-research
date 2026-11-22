"""Vera: how often does the 3-group seed-level t-interval for D_a exclude 0 when N and C are exchangeable?

Shuffles the N/C label among followers... no: among ALL runs within template x seed group (as vera_null.py), recomputes
announce|follow per cell per seed group, D_a per group, and the t-interval (2 df) over the 3 group values.
A narrow interval that excludes 0 under shuffles shows the seed-level interval is not evidence of a real rise.
Usage: python3 vera_seed_null.py rebuild.json
"""
import json, sys
import numpy as np
from statistics import mean, stdev

T2 = 4.302652729911275  # t(0.975, 2)


def seed(r):
    return int(r["file"].split("/")[-1].split(".")[0]) % 3


def da_groups(rows, lab):
    vals = []
    for s in range(3):
        fn = [r for r, l in zip(rows, lab) if l == 0 and seed(r) == s and r["follow"]]
        fc = [r for r, l in zip(rows, lab) if l == 1 and seed(r) == s and r["follow"]]
        if not fn or not fc:
            return None
        vals.append(sum(r["announce"] for r in fn) / len(fn) - sum(r["announce"] for r in fc) / len(fc))
    return vals


def ci(v):
    m, sd = mean(v), stdev(v)
    h = T2 * sd / 3 ** 0.5
    return m - h, m + h


d = json.load(open(sys.argv[1]))
rows = d["rows"]["N"] + d["rows"]["C"]
lab = np.array([0] * len(d["rows"]["N"]) + [1] * len(d["rows"]["C"]))
obs = da_groups(rows, lab)
print("observed groups", [round(x, 4) for x in obs], "t-CI", [round(x, 4) for x in ci(obs)])
strata = {}
for j, r in enumerate(rows):
    strata.setdefault((r["template"], seed(r)), []).append(j)
rng = np.random.default_rng(11)
B, excl, narrow = 20000, 0, 0
w_obs = ci(obs)[1] - ci(obs)[0]
for _ in range(B):
    l2 = lab.copy()
    for idx in strata.values():
        l2[idx] = rng.permutation(lab[idx])
    v = da_groups(rows, l2)
    lo, hi = ci(v)
    excl += (lo > 0) or (hi < 0)
    narrow += (hi - lo) <= w_obs
print(f"shuffles {B}: seed-level t-CI excludes 0 in {excl / B:.4f}; width <= observed {w_obs:.4f} in {narrow / B:.4f}")
json.dump(dict(observed_groups=obs, observed_ci=ci(obs), shuffles=B, p_excludes_0=excl / B, p_width_le_obs=narrow / B),
          open("seed_null.json", "w"), indent=1)
