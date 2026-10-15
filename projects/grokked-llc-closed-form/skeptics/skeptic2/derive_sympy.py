"""skeptic2: symbolic checks of the per-neuron count and the capacity cap.

Model on one-hot pair inputs x=[e_a; e_b]: f_c(a,b) = sum_j V[c,j] * (u_j[a] + s_j[b])**2,
with w_j = (u_j, s_j) in R^{2p}. Loss sees only centred predictions on training pairs.

Checks:
 1. One neuron, p=3, full data, symbolic generic point: null space of the Jacobian of the
    centred outputs has dim exactly 2: scaling (w -> t w, v -> v/t^2) and shift (u+c, s-c).
    So per-neuron rank = 3p - 2, not d + p - 1 = 3p - 1.
 2. Identity x^T Q x = Q_aa + Q_{p+b,p+b} + 2 Q_{a,p+b} (sympy).
 3. Paper's over-param value p d(d+1)/4 with d=2p exceeds the hard cap p(N-1)/2 for all
    N <= p^2 (sympy inequality as a polynomial in p).
"""
import itertools
import random

import sympy as sp

p = 3
# rational generic point (exact arithmetic, no float rank issues)
random.seed(0)
rnd = lambda: sp.Rational(random.randint(-9, 9) or 1, random.randint(1, 7))
u = sp.symbols(f"u0:{p}")
s = sp.symbols(f"s0:{p}")
v = sp.symbols(f"v0:{p}")
params = list(u) + list(s) + list(v)
pairs = list(itertools.product(range(p), range(p)))
outs = []
for c in range(p):
    col = [v[c] * (u[a] + s[b]) ** 2 for a, b in pairs]
    mean = sum(col) / len(col)
    outs += [e - mean for e in col]  # centring over samples, per output
J = sp.Matrix(outs).jacobian(params)
point = {x: rnd() for x in params}
Jp = J.subs(point)
r = Jp.rank()
ns = Jp.nullspace()
print(f"[1] p={p}: params={len(params)}, rank={r}, 3p-2={3*p-2}, 3p-1={3*p-1}, nullity={len(ns)}")
assert r == 3 * p - 2 and len(ns) == 2
# the two symmetry generators are in the null space
shift = sp.Matrix([1] * p + [-1] * p + [0] * p)
scale = sp.Matrix([point[x] for x in u] + [point[x] for x in s] + [-2 * point[x] for x in v])
assert (Jp * shift).is_zero_matrix and (Jp * scale).is_zero_matrix
print("    shift (u+c,s-c) and scaling (w*t, v/t^2) generate the null space")

# 2. quadratic form identity
pp = sp.Integer(4)
Q = sp.Matrix(2 * 4, 2 * 4, lambda i, j: sp.Symbol(f"q{min(i,j)}_{max(i,j)}"))
ok = True
for a in range(4):
    for b in range(4):
        x = sp.zeros(8, 1); x[a] = 1; x[4 + b] = 1
        ok &= sp.simplify((x.T * Q * x)[0] - (Q[a, a] + Q[4 + b, 4 + b] + 2 * Q[a, 4 + b])) == 0
print(f"[2] x^T Q x = Q_aa + Q_(p+b)(p+b) + 2 Q_a(p+b) for p=4: {ok}")
assert ok

# 3. over-param formula vs cap: p d(d+1)/4 - p(N-1)/2 with N = p^2 (largest N) is > 0 for p >= 2
P = sp.symbols("p", positive=True, integer=True)
d = 2 * P
gap = sp.expand(P * d * (d + 1) / 4 - P * (P**2 - 1) / 2)
print(f"[3] p d(d+1)/4 - p(p^2-1)/2 = {sp.factor(gap)}  (positive for all p>=1)")
assert sp.simplify(gap - P * (P**2 + P + 1) / 2) == 0  # = p(p^2+p+1)/2 > 0
for pv in (7, 11, 17, 23, 31):
    print(f"    p={pv}: paper over-param {pv*2*pv*(2*pv+1)/4:.1f}  vs full-data cap {pv*(pv*pv-1)/2:.1f}")
print("ALL SYMPY CHECKS PASS")
