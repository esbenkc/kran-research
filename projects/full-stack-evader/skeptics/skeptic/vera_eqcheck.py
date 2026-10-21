"""Vera: own checks of the post's displayed equations (sympy + numpy).
E1  KL(p || softmax(z + W d)) ~= 1/2 d' F d,  F = W'(diag p - p p')W   (numeric, shrinking d)
E2  min 1/2 d'F d  s.t.  w'd = g   ->   g^2 / (2 w'F^+ w),  d* = g F^+ w / (w'F^+ w)   (sympy Lagrange, symbolic 2x2;
    numeric 6x6 PSD rank-deficient with w in range(F); random feasible moves never lower)
E3  the loss: 1/8 sum (s(f_i)-s(t_i))^2 + lam/2 (KL_c + KL_w) is minimised at s(f_i) = s(t_i), KL = 0 (sympy gradient)
    python vera_eqcheck.py
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(0)
out = {}
# E1
V, d = 50, 6
W = rng.standard_normal((V, d)); z = rng.standard_normal(V)
p = np.exp(z - z.max()); p /= p.sum()
F = W.T @ (np.diag(p) - np.outer(p, p)) @ W
for eps in [1e-1, 1e-2, 1e-3]:
    dl = eps * rng.standard_normal(d)
    q = np.exp(z + W @ dl - (z + W @ dl).max()); q /= q.sum()
    kl = (p * (np.log(p) - np.log(q))).sum()
    out[f"E1_ratio_eps{eps:g}"] = kl / (0.5 * dl @ F @ dl)
# E2 symbolic
a, b, c, w1, w2, g, l = sp.symbols("a b c w1 w2 g l", real=True)
x1, x2 = sp.symbols("x1 x2", real=True)
Fs = sp.Matrix([[a, b], [b, c]]); ws = sp.Matrix([w1, w2]); xs = sp.Matrix([x1, x2])
L = (xs.T * Fs * xs)[0] / 2 - l * ((ws.T * xs)[0] - g)
sol = sp.solve([sp.diff(L, x1), sp.diff(L, x2), (ws.T * xs)[0] - g], [x1, x2, l], dict=True)[0]
kmin = sp.simplify(((xs.T * Fs * xs)[0] / 2).subs(sol))
target = sp.simplify(g ** 2 / (2 * (ws.T * Fs.inv() * ws)[0]))
out["E2_sympy_equal"] = bool(sp.simplify(kmin - target) == 0)
# E2 numeric, rank-deficient F, w in range(F)
A = rng.standard_normal((6, 4)); Fn = A @ A.T  # rank 4
w = Fn @ rng.standard_normal(6); gg = 1.7
Fp = np.linalg.pinv(Fn)
dstar = gg * Fp @ w / (w @ Fp @ w)
out["E2_num_constraint"] = float(w @ dstar - gg)
out["E2_num_kl_vs_formula"] = float(0.5 * dstar @ Fn @ dstar - gg ** 2 / (2 * w @ Fp @ w))
low = np.inf
for _ in range(20000):
    r = rng.standard_normal(6); r -= (w @ r) / (w @ w) * w
    dd = dstar + r * rng.uniform(0, 2)
    low = min(low, 0.5 * dd @ Fn @ dd)
out["E2_random_feasible_min_minus_formula"] = float(low - gg ** 2 / (2 * w @ Fp @ w))
# E2 caveat: if w has a component in null(F), the second-order cost of moving the score by g is 0 (bound is infinite)
nul = np.linalg.svd(Fn)[0][:, -1]  # eigenvector of a zero eigenvalue
w2 = w + nul
dn = gg * nul / (w2 @ nul)
out["E2_null_component_cost"] = float(0.5 * dn @ Fn @ dn)
out["E2_null_component_shift"] = float(w2 @ dn)
# E3
s = sp.symbols("sf1:9"), sp.symbols("st1:9")
lam, kc, kw = sp.symbols("lam kc kw", nonnegative=True)
Ls = sum((s[0][i] - s[1][i]) ** 2 for i in range(8)) / 8 + lam / 2 * (kc + kw)
grad = [sp.diff(Ls, s[0][i]) for i in range(8)]
out["E3_grad_zero_at_equal_twins"] = all(sp.simplify(gi.subs({s[0][i]: s[1][i] for i in range(8)})) == 0 for gi in grad)
for k, v in out.items():
    print(k, v)
