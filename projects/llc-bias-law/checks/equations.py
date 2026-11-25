"""Exact / numeric checks of every displayed equation of llc-bias-law (Iris, research).

Run:  uv run --with sympy --with numpy python checks/equations.py   (from the project dir)
Each check prints PASS or raises. No data files are read; nothing here is a scored cell.
"""
from fractions import Fraction as F
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))
from rrr_lambda import rrr_lambda  # noqa: E402  (Aoyagi & Watanabe 2005; parent run's code)


def ok(name):
    print(f"PASS  {name}")


# E1. Aoyagi values used in the post and PREREG
vals = {(32, 32, 16, 1): 236, (32, 32, 16, 8): 312, (24, 24, 12, 3): 152, (24, 24, 12, 6): F(351, 2),
        (48, 48, 8, 2): F(455, 2), (20, 20, 10, 5): 122, (40, 24, 12, 4): 200}
for k, v in vals.items():
    assert rrr_lambda(*k) == v, (k, rrr_lambda(*k), v)
ok("E1 Aoyagi lambda for the 2 training + 5 blind shapes")

# E2. Split identity of the closed form: lambda(M,N,H,r) = r(M+N-r)/2 + lambda(M-r,N-r,H-r,0), on a box
n = 0
for M in range(1, 17):
    for N in range(1, 17):
        for H in range(1, 17):
            for r in range(0, min(M, N, H) + 1):
                rest = rrr_lambda(M - r, N - r, H - r, 0) if min(M - r, N - r, H - r) > 0 else 0
                assert rrr_lambda(M, N, H, r) == F(r * (M + N - r), 2) + rest, (M, N, H, r)
                n += 1
ok(f"E2 split identity on the box M,N,H <= 16 ({n} cases, 0 mismatches)")

# E3. ULA (SGLD without gradient noise) on a quadratic direction with a tether:
#     x' = x - (eps/2) a x + sqrt(eps) xi  ->  stationary Var = 1 / (a (1 - eps a / 4))
eps, a, h, nb, g, s = sp.symbols("epsilon a h nbeta gamma s", positive=True)
V = sp.symbols("V", positive=True)
Vsol = sp.solve(sp.Eq(V, (1 - eps * a / 2) ** 2 * V + eps), V)[0]
assert sp.simplify(Vsol - 1 / (a * (1 - eps * a / 4))) == 0
ok("E3 stationary variance 1/(a(1 - eps a/4))")

# E4. Per-direction estimate: nbeta * (h/2) * Var with a = nbeta h + gamma equals (1/2) [1/(1+g)] / [1 - c/4],
#     g = gamma/(nbeta h), c = eps (nbeta h + gamma)
lam_dir = nb * h / 2 * Vsol.subs(a, nb * h + g)
gg = g / (nb * h)
cc = eps * (nb * h + g)
assert sp.simplify(lam_dir - sp.Rational(1, 2) / (1 + gg) / (1 - cc / 4)) == 0
ok("E4 regular-direction estimate (1/2)[1/(1+g)]/[1-c/4]")

# E5. Scaling under w -> s w: eps ~ s^2, gamma ~ s^-2, h ~ s^-2, nbeta ~ 1, k4 ~ s^-4
k4 = sp.symbols("k4", positive=True)
scale = {eps: eps * s**2, g: g * s**-2, h: h * s**-2, k4: k4 * s**-4}
def fac(expr):
    return sp.simplify(expr.subs(scale, simultaneous=True) / expr)
assert fac(eps * nb / g) == s**4
for e in (eps * g, eps * nb * h, g / (nb * h), nb * k4 / g**2):
    assert fac(e) == 1, e
ok("E5 eps*nbeta/gamma scales as s^4; eps*gamma, eps*nbeta*h, gamma/(nbeta*h), q = nbeta*k4/gamma^2 are scale-free")

# E6. Null-block rescaling (Kasper #30691): with K0 = 1/2 ||B A||^2 (homogeneous of degree 4) and w = nbeta^(-1/4) u,
#     nbeta K0(w) + gamma/2 ||w||^2 = 1/2 ||B_u A_u||^2 + (gamma / sqrt(nbeta)) / 2 ||u||^2.  Checked on a 2x2x2 block.
Bu = sp.Matrix(2, 2, sp.symbols("b0:4"))
Au = sp.Matrix(2, 2, sp.symbols("a0:4"))
t = nb ** sp.Rational(-1, 4)
def sq(Mx):
    return sum(x**2 for x in Mx)
lhs = nb * sq((t * Bu) * (t * Au)) / 2 + g / 2 * (sq(t * Bu) + sq(t * Au))
rhs = sq(Bu * Au) / 2 + (g / sp.sqrt(nb)) / 2 * (sq(Bu) + sq(Au))
assert sp.simplify(sp.expand(lhs - rhs)) == 0
ok("E6 null-block rescaling: depends on (nbeta, gamma) only through gamma/sqrt(nbeta) (= q^(-1/2) up to k4)")

# E7. Strong-spring (Gaussian) limit: A, B entries iid N(0, 1/gamma) ->
#     E[nbeta * 1/2 ||B A||^2] = nbeta M'N'H' / (2 gamma^2) = q M'N'H'  with q = nbeta k4/gamma^2, k4 = 1/2.
#     Exact: E[(sum_k B_ik A_kj)^2] = H'/gamma^2; summed over i, j. Monte Carlo check on (5,4,3).
rng = np.random.default_rng(0)
Mp, Np, Hp, gam, nbv = 5, 4, 3, 2.0, 7.0
Bs = rng.standard_normal((200000, Np, Hp)) / np.sqrt(gam)
As = rng.standard_normal((200000, Hp, Mp)) / np.sqrt(gam)
mc = nbv * 0.5 * np.mean(np.sum((Bs @ As) ** 2, axis=(1, 2)))
exact = nbv * Mp * Np * Hp / (2 * gam**2)
q = nbv * 0.5 / gam**2
assert abs(exact - q * Mp * Np * Hp) < 1e-12
assert abs(mc / exact - 1) < 0.01, (mc, exact)
ok(f"E7 Gaussian limit nbeta*E[K0] = q*M'N'H' (MC {mc:.3f} vs exact {exact:.3f})")

# E8. z/q slopes M'N'H'/lambda0 (training and blind shapes), as quoted in PREREG S4 and the room
def slope(M, N, H, r):
    l0 = rrr_lambda(M - r, N - r, H - r, 0)
    return F((M - r) * (N - r) * (H - r)) / l0
assert abs(float(slope(32, 32, 16, 1)) - 70.49) < 0.01 and abs(float(slope(32, 32, 16, 8)) - 52.36) < 0.01
ok("E8 z/q slopes: train 70.49 (r1), 52.36 (r8)")

# E9. Named recipe S2 mapping (Lau et al. 2308.12108 App. J.1, '1k' row): n = 1e5, nbeta = n/ln n,
#     inputs uniform[-10,10] -> variance 100/3, nbeta_eff = nbeta * 100/3, gamma 1 -> q = nbeta_eff/2
x = sp.symbols("x")
var_u = sp.integrate(x**2, (x, -10, 10)) / 20
assert var_u == sp.Rational(100, 3)
nbeta_lau = 1e5 / np.log(1e5)
nbeta_eff = nbeta_lau * 100 / 3
assert abs(nbeta_lau - 8685.9) < 0.1 and abs(nbeta_eff - 2.8953e5) < 50
ok(f"E9 Lau recipe: nbeta {nbeta_lau:.1f}, nbeta_eff {nbeta_eff:.4g}, q {nbeta_eff * 0.5:.3g}")

# E10. Parent gate row S2b: n = 128 -> nbeta = 128/ln 128 = 26.38; gamma 10 -> q = nbeta/2/100
nbp = 128 / np.log(128)
assert abs(nbp - 26.38) < 0.01 and abs(nbp * 0.5 / 100 - 0.132) < 0.001
ok(f"E10 parent gate: nbeta {nbp:.2f}, q at gamma 10 = {nbp * 0.5 / 100:.3f}")

print("ALL PASS")
