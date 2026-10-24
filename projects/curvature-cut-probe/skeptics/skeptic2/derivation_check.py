"""Hugo (skeptic2): symbolic + numeric checks of every displayed step of the floor lemma.

(1) sympy: Hessian of x -> KL(p || softmax(l + W x)) at x = 0 equals W^T (diag p - p p^T) W (V=3, d=2).
(2) sympy: min_a a^2/(2 gf) + (D-a)^2/(2 gt) = D^2 / (2 (gf + gt)) at a = D gf/(gf+gt).
(3) numeric toy (V=300, d=24, peaked p): exact constrained minimum of KL s.t. u.x = -D (scipy SLSQP,
    convex problem) vs D^2/(2 u'F^+u) for small and large D; and the kernel case (u with a component in
    ker F after making W rank-deficient) gives a floor -> 0.
(4) eps-regularised quad form u'(F+eps I)^-1 u <= u'F^+u (so the eps floor D^2/(2 u'(F+eps I)^-1 u) sits slightly
    ABOVE the pseudo-inverse floor; gap <= 1e-4 relative at eps <= 1e-6 lambda_max in the toy).
Note from (3): at large D the exact minimum falls BELOW the second-order formula (0.955x at D=1 in the toy):
the formula is the small-move limit, not a lower bound; only the exact minimum or a certificate is a floor.
"""
import numpy as np
import sympy as sp
from scipy.optimize import minimize

# (1)
x1, x2 = sp.symbols('x1 x2')
l = sp.Matrix(sp.symbols('l0:3'))
W = sp.Matrix(3, 2, sp.symbols('w0:6'))
x = sp.Matrix([x1, x2])
lz = l + W * x
Z0 = sum(sp.exp(l[i]) for i in range(3))
p = [sp.exp(l[i]) / Z0 for i in range(3)]
logq = [lz[i] - sp.log(sum(sp.exp(lz[j]) for j in range(3))) for i in range(3)]
KL = sum(p[i] * (sp.log(p[i]) - logq[i]) for i in range(3))
H = sp.hessian(KL, (x1, x2)).subs({x1: 0, x2: 0})
P = sp.Matrix(p)
F = W.T * (sp.diag(*p) - P * P.T) * W
diff = sp.simplify(H - F)
grad0 = sp.simplify(sp.Matrix([sp.diff(KL, v) for v in (x1, x2)]).subs({x1: 0, x2: 0}))
print('(1) Hessian - W^T(diag p - pp^T)W =', diff.T, '| gradient at 0 =', grad0.T)

# (2)
a, D, gf, gt = sp.symbols('a D g_f g_t', positive=True)
f = a ** 2 / (2 * gf) + (D - a) ** 2 / (2 * gt)
astar = sp.solve(sp.diff(f, a), a)[0]
print('(2) a* =', sp.simplify(astar), '| min - D^2/(2(gf+gt)) =', sp.simplify(f.subs(a, astar) - D ** 2 / (2 * (gf + gt))))

# (3)
rng = np.random.default_rng(0)
V, d = 300, 24
Wn = rng.standard_normal((V, d))
l0 = 3.0 * rng.standard_normal(V)
l0[0] += 12  # peaked
p0 = np.exp(l0 - l0.max()); p0 /= p0.sum()
Fn = Wn.T @ (np.diag(p0) - np.outer(p0, p0)) @ Wn
u = rng.standard_normal(d)


def kl(xv, Wm=Wn):
    lq = l0 + Wm @ xv
    lq = lq - lq.max()
    lq = lq - np.log(np.exp(lq).sum())
    lp = np.log(p0)
    return float((p0 * (lp - lq)).sum())


for Dv in (1e-3, 1e-2, 1e-1, 1.0):
    formula = Dv ** 2 / (2 * u @ np.linalg.pinv(Fn) @ u)
    r = minimize(kl, np.zeros(d), method='SLSQP', constraints=[{'type': 'eq', 'fun': lambda xv: u @ xv + Dv}],
                 options={'ftol': 1e-16, 'maxiter': 500})
    print('(3) D=%g exact %.6e formula %.6e ratio %.4f' % (Dv, r.fun, formula, r.fun / formula))
# kernel case: W rank-deficient (last column = 0): u with a component on e_d is free
Wk = Wn.copy(); Wk[:, -1] = 0
Fk = Wk.T @ (np.diag(p0) - np.outer(p0, p0)) @ Wk
uk = u.copy()
r = minimize(lambda xv: kl(xv, Wk), np.zeros(d), method='SLSQP',
             constraints=[{'type': 'eq', 'fun': lambda xv: uk @ xv + 1.0}], options={'ftol': 1e-16, 'maxiter': 500})
print('(3) kernel case: exact min %.3e (floor 0 expected); |x_d| = %.3g' % (r.fun, abs(r.x[-1])))

# (4)
lam, Q = np.linalg.eigh(Fn)
ur = Q @ (Q.T @ u)
for e in (1e-10, 1e-8, 1e-6):
    qe = u @ np.linalg.solve(Fn + e * lam[-1] * np.eye(d), u)
    print('(4) eps %.0e: u(F+epsI)^-1u = %.6e vs u F^+ u = %.6e (eps version <= : %s)' % (
        e, qe, u @ np.linalg.pinv(Fn) @ u, qe <= u @ np.linalg.pinv(Fn) @ u + 1e-12))
