"""Checks for the two displayed equations of the curvature-cut-probe post (expert foldout).

Eq. 1: KL(p || softmax(Wz + W d)) = 1/2 d' F d + O(|d|^3),  F = W'(diag p - p p')W.
  - sympy: the Hessian of KL at d = 0 equals F exactly (small symbolic case), gradient is 0.
  - numeric: on a random W (V=500, n=24) the ratio KL / (1/2 d'Fd) -> 1 as |d| -> 0, error ~ |d|.
Eq. 2: min over a split D = d_f + d_t of d_f^2/(2 g_f) + d_t^2/(2 g_t) equals D^2 / (2 (g_f + g_t)),
  where d^2/(2g) with g = a' F^+ a is the cheapest second-order move shifting a' z by d.
  - sympy: closed-form minimisation of the split.
  - numeric: the minimum of 1/2 d'Fd subject to a'd = D equals D^2 / (2 a'F^+ a) (Lagrange), checked against
    scipy's constrained solver and the explicit minimiser d* = D F^+ a / (a'F^+ a).
Run: uv run python equation_checks.py   (prints PASS lines; exits 1 on any failure)
"""
import sys

import numpy as np
import sympy as sp

fails = 0


def check(name, ok, detail):
    global fails
    print(("PASS " if ok else "FAIL ") + name + " | " + detail)
    fails += 0 if ok else 1


# ---------- Eq. 1, symbolic (V = 3 words, n = 2 dims) ----------
V, n = 3, 2
Wsym = sp.Matrix(V, n, lambda i, j: sp.Symbol(f"w{i}{j}", real=True))
l0 = sp.Matrix(V, 1, lambda i, j: sp.Symbol(f"l{i}", real=True))
d = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"d{i}", real=True))
p = [sp.exp(l0[i]) / sum(sp.exp(l0[k]) for k in range(V)) for i in range(V)]
l1 = l0 + Wsym * d
logZ1 = sp.log(sum(sp.exp(l1[k]) for k in range(V)))
logZ0 = sp.log(sum(sp.exp(l0[k]) for k in range(V)))
# KL(p || q) = sum p (log p - log q) = sum p (l0 - l1) + logZ1 - logZ0
KL = sum(p[i] * (l0[i] - l1[i]) for i in range(V)) + logZ1 - logZ0
at0 = {d[i]: 0 for i in range(n)}
grad0 = [sp.simplify(sp.diff(KL, d[i]).subs(at0)) for i in range(n)]
check("eq1 symbolic gradient at d=0 is 0", all(g == 0 for g in grad0), str(grad0))
P = sp.Matrix(p)
F = Wsym.T * (sp.diag(*p) - P * P.T) * Wsym
H = sp.Matrix(n, n, lambda i, j: sp.diff(KL, d[i], d[j]).subs(at0))
diff = sp.simplify(H - F)
check("eq1 symbolic Hessian at d=0 equals W'(diag p - pp')W", diff == sp.zeros(n, n), str(diff))

# ---------- Eq. 1, numeric ----------
rng = np.random.default_rng(0)
Vn, nn = 500, 24
W = rng.normal(size=(Vn, nn))
z = rng.normal(size=nn)
logits = W @ z
p0 = np.exp(logits - logits.max()); p0 /= p0.sum()
Fn = W.T @ (np.diag(p0) - np.outer(p0, p0)) @ W
u = rng.normal(size=nn); u /= np.linalg.norm(u)
rows = []
for eps in [1e-1, 1e-2, 1e-3]:
    dd = eps * u
    l1n = logits + W @ dd
    q = np.exp(l1n - l1n.max()); q /= q.sum()
    kl = float(np.sum(p0 * (np.log(p0) - np.log(q))))
    quad = 0.5 * dd @ Fn @ dd
    rows.append((eps, kl / quad))
errs = [abs(r - 1) for _, r in rows]
check("eq1 numeric KL/(1/2 d'Fd) -> 1 linearly in |d|", errs[2] < 2e-3 and errs[1] < errs[0] and errs[2] < errs[1],
      ", ".join(f"|d|={e:g}: {r:.6f}" for e, r in rows))

# ---------- Eq. 2, symbolic split ----------
Dd, gf, gt, x = sp.symbols("D g_f g_t x", positive=True)
cost = x**2 / (2 * gf) + (Dd - x)**2 / (2 * gt)
xs = sp.solve(sp.diff(cost, x), x)[0]
cmin = sp.simplify(cost.subs(x, xs))
check("eq2 symbolic: min split cost = D^2/(2(g_f+g_t))", sp.simplify(cmin - Dd**2 / (2 * (gf + gt))) == 0, str(cmin))

# ---------- Eq. 2 building block, numeric: min 1/2 d'Fd s.t. a'd = D ----------
from scipy.optimize import minimize

a = rng.normal(size=nn)
Dn = 0.7
eps_reg = 1e-8 * np.linalg.eigvalsh(Fn).max()
Fr = Fn + eps_reg * np.eye(nn)
g = a @ np.linalg.solve(Fr, a)
closed = Dn**2 / (2 * g)
dstar = Dn * np.linalg.solve(Fr, a) / g
res = minimize(lambda v: 0.5 * v @ Fr @ v, np.zeros(nn), jac=lambda v: Fr @ v,
               constraints=[{"type": "eq", "fun": lambda v: a @ v - Dn, "jac": lambda v: a}],
               method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
check("eq2 numeric: constrained min of 1/2 d'Fd = D^2/(2 a'F^+ a)",
      abs(res.fun - closed) / closed < 1e-4 and abs(0.5 * dstar @ Fr @ dstar - closed) / closed < 1e-10,
      f"solver {res.fun:.6e}, closed {closed:.6e}, explicit d* {0.5 * dstar @ Fr @ dstar:.6e}")

sys.exit(1 if fails else 0)
