"""Vera (skeptic): sympy + numeric checks of the two lemmas the bound is built from.

L1 (direction move): sup_{||delta||<=r} || (x+delta)/||x+delta|| - x/||x|| || = chord(rho), rho = r/||x||,
    chord(rho) = 2 sin(asin(rho)/2) = sqrt(2 - 2 sqrt(1-rho^2)) for rho < 1, and 2 for rho >= 1.
    Also: chord(rho) >= rho, so the first-order factor rho = r/||x|| UNDER-states the move (ratio -> sqrt2 at rho=1).
L2 (softmax): TV(softmax(z), softmax(z+delta)) <= tanh((max delta - min delta)/4), tight for two groups.
"""
import math
import numpy as np
import sympy as sp

rho, c = sp.symbols("rho c", positive=True)
chord_expr = sp.sqrt(2 - 2 * sp.sqrt(1 - rho**2))
# 2 sin(asin(rho)/2) == sqrt(2-2 sqrt(1-rho^2)) on (0,1)
diff = sp.simplify((2 * sp.sin(sp.asin(rho) / 2))**2 - chord_expr**2)
print("L1 identity 2sin(asin/2)^2 - chord^2 =", diff)
# chord >= rho  <=>  2 - 2c >= 1 - c^2 with c = sqrt(1-rho^2)  <=>  (1-c)^2 >= 0
print("L1 chord^2 - rho^2 with c:", sp.factor((2 - 2 * c) - (1 - c**2)))
print("L1 chord/rho at rho=1:", sp.nsimplify(chord_expr.subs(rho, 1) / 1), " at rho->0:", sp.limit(chord_expr / rho, rho, 0))

rng = np.random.default_rng(0)
# numeric sup of L1 by sampling + boundary search
worst = 0
for _ in range(2000):
    D = rng.integers(2, 64)
    x = rng.normal(size=D) * rng.uniform(0.1, 10)
    rr = rng.uniform(0, 1.5) * np.linalg.norm(x)
    p = rr / np.linalg.norm(x)
    ch = 2 * math.sin(math.asin(p) / 2) if p < 1 else 2.0
    # best delta: rotate x away by max angle; sample many deltas on the sphere
    dl = rng.normal(size=(256, D))
    dl = rr * dl / np.linalg.norm(dl, axis=1, keepdims=True)
    # analytic maximiser for p<1: delta perpendicular-ish with angle asin(p)
    xn = x / np.linalg.norm(x)
    v = rng.normal(size=D); v -= v @ xn * xn; v /= np.linalg.norm(v)
    if p < 1:
        th = math.asin(p)
        y = math.cos(th) * xn + math.sin(th) * v      # tangent point
        dstar = y * math.sqrt(np.linalg.norm(x)**2 - rr**2) - x
        dl = np.vstack([dl, dstar])
    m = np.linalg.norm((x + dl) / np.linalg.norm(x + dl, axis=1, keepdims=True) - xn, axis=1).max()
    worst = max(worst, m / ch)
print(f"L1 numeric max(achieved/chord) over 2000 cases: {worst:.6f} (must be <= 1, ~1 = tight)")

worst2, tight = 0, 0
for _ in range(20000):
    T = rng.integers(2, 40)
    z = rng.normal(size=T) * rng.uniform(0.1, 8)
    dz = rng.normal(size=T) * rng.uniform(0.01, 6)
    p = np.exp(z - z.max()); p /= p.sum()
    q = np.exp(z + dz - (z + dz).max()); q /= q.sum()
    tv = 0.5 * np.abs(p - q).sum()
    b = math.tanh((dz.max() - dz.min()) / 4)
    worst2 = max(worst2, tv / b)
for o in [0.1, 1, 4, 10]:  # tight case: two groups, a = 1/(1+e^{o/2})
    a = 1 / (1 + math.exp(o / 2))
    pa = a * math.exp(o) / (a * math.exp(o) + 1 - a)
    tight = max(tight, (pa - a) / math.tanh(o / 4))
print(f"L2 numeric max(TV/tanh(osc/4)) over 20000 random cases: {worst2:.6f}; two-group case ratio: {tight:.6f}")
