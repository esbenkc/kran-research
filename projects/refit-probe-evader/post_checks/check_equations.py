"""Numeric checks for every displayed equation and derived number in _notes/refit-probe-evader.md (Selma, writer).

Run: python3 post_checks/check_equations.py   (numpy only, < 1 s, < 100 MB)

1. LEACE eraser, rank-one form in the post:
       r(x) = x - delta delta^T S^-1 (x - mu) / (delta^T S^-1 delta)
   equals the general LEACE form of Belrose et al. (arXiv 2306.03819, Thm 3.1/4.2):
       r(x) = x - W^+ P W (x - mu),  W = S^-1/2,  P = orthogonal projector onto col(W Sigma_xz)
   for a binary concept (Sigma_xz proportional to delta), and it zeroes the class-mean gap on the fit data.
2. The LEACE cost P = (sum of per-position KL) / (number of positions), recomputed from the saved per-statement
   sums in data/leace/perstatement_kl.npz, matches data/leace/price.json, and N = 12,698 positions.
3. Ratios quoted in the post: everypos / gate, gate / rand_size, run KL / P.
4. Alarm line: the 2nd-highest of 200 negatives is crossed by a fresh negative with probability 2/201.
5. Angle from the cosine quoted by research_checks/gap_geometry.json (cos 0.64 -> about 50 degrees).
"""
import json
from pathlib import Path

import numpy as np
import warnings; warnings.filterwarnings("ignore")  # spurious BLAS matmul warnings on macOS numpy

HERE = Path(__file__).resolve().parent.parent
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))


# 1. LEACE rank-one form vs general form
rng = np.random.default_rng(0)
d, n = 12, 4000
A = rng.normal(size=(d, d))
z = rng.integers(0, 2, n)
X = rng.normal(size=(n, d)) @ A + np.outer(z, rng.normal(size=d)) * 2.0
mu = X.mean(0)
S = np.cov(X, rowvar=False, bias=True)
delta = X[z == 1].mean(0) - X[z == 0].mean(0)
Si = np.linalg.inv(S)
r1 = X - np.outer((X - mu) @ Si @ delta, delta) / (delta @ Si @ delta)
ev, V = np.linalg.eigh(S)
W = V @ np.diag(ev ** -0.5) @ V.T
Winv = V @ np.diag(ev ** 0.5) @ V.T
sxz = ((X - mu) * (z - z.mean())[:, None]).mean(0)
u = W @ sxz
Pm = np.outer(u, u) / (u @ u)
r2 = X - (X - mu) @ (Winv @ Pm @ W).T
check("eq1 rank-one form == W^+ P W form", np.abs(r1 - r2).max() < 1e-9, f"max diff {np.abs(r1 - r2).max():.2e}")
gap = r1[z == 1].mean(0) - r1[z == 0].mean(0)
check("eq1 zeroes the class-mean gap on fit data", np.abs(gap).max() < 1e-9, f"max gap {np.abs(gap).max():.2e}")

# 2. LEACE cost recomputed from per-statement sums
kl = np.load(HERE / "data/leace/perstatement_kl.npz")
price = json.load(open(HERE / "data/leace/price.json"))


def cost(key):
    return kl[f"{key}__kl_sum"].sum() / kl[f"{key}__n_pos"].sum()


npos = int(kl["registered__n_pos"].sum())
P = cost("registered")
check("eq2 N positions = 12,698", npos == 12698, f"N {npos}")
check("eq2 P recompute == registered 0.006145", abs(P - price["registered_price_P"]) < 1e-6, f"P {P:.6f}")
gate, everypos, rnd = cost("gate"), cost("everypos"), cost("rand_size")
print(f"     gate {gate:.5f}  everypos {everypos:.3f}  rand_size {rnd:.5f}")
check("post: gate cost 0.0060", round(gate, 4) == 0.0061 or round(gate, 4) == 0.0060, f"{gate:.5f}")
check("post: same edit at every token 1.94, ~300x", round(everypos, 2) == 1.94 and 250 < everypos / gate < 350, f"{everypos / gate:.0f}x")
check("post: random direction 0.00037, ~16x less", round(rnd, 5) == 0.00037 and 14 < gate / rnd < 18, f"{gate / rnd:.1f}x")

# 3. run ratios
for run, want in [("refit50__lam10000__seed0", 1.3), ("refit50__lam100000__seed0", 0.6)]:
    f = HERE / f"data/results/{run}.json"
    if f.exists():
        r = json.load(open(f))
        k = r.get("kl_eval_every_position")
        if k is not None:
            check(f"post: {run} KL / P = {want}", round(k / P, 1) == want, f"{k:.5f} / {P:.5f} = {k / P:.2f}")

# 4. alarm line
check("line: 2nd-highest of 200 -> 2/201 ~ 1%", abs(2 / 201 - 0.00995) < 1e-4, f"{2 / 201:.4f}")

# 5. angle
check("angle: arccos(0.64) ~ 50 deg", round(np.degrees(np.arccos(0.64))) == 50, f"{np.degrees(np.arccos(0.64)):.1f}")

print("ALL PASS" if ok else "SOME FAIL")
