"""Symbolic + numeric checks for every displayed equation of the lemma.

Run: uv run python projects/probe-evasion-geometry/theory/check_equations.py
Each check prints PASS/FAIL with the max error. Exit code 1 on any FAIL.
"""

import sys

import numpy as np
import sympy as sp
from scipy.stats import norm

rng = np.random.default_rng(0)
FAILS = []


def check(name, err, tol=1e-8):
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}  (max err {err:.2e})")
    if not ok:
        FAILS.append(name)


def rand_spd(d):
    a = rng.normal(size=(d, d))
    lam = 1.0 / np.arange(1, d + 1)
    q, _ = np.linalg.qr(a)
    return q @ np.diag(lam) @ q.T


def dprime(u, delta, S):
    return u @ delta / np.sqrt(u @ S @ u)


def cos_S(u, w, S):
    return u @ S @ w / np.sqrt((u @ S @ u) * (w @ S @ w))


# E1. AUROC of a linear probe on equal-covariance Gaussians = Phi(d'/sqrt 2).
# Score difference s1 - s0 ~ N(u.Delta, 2 u'Su); AUROC = P(s1 > s0).
d = 8
S = rand_spd(d)
delta = rng.normal(size=d)
u = rng.normal(size=d)
L = np.linalg.cholesky(S)
n = 400_000
x0 = rng.normal(size=(n, d)) @ L.T
x1 = rng.normal(size=(n, d)) @ L.T + delta
auc_mc = np.mean((x1 @ u) > (x0 @ u))
auc_th = norm.cdf(dprime(u, delta, S) / np.sqrt(2))
check("E1 AUROC = Phi(d'/sqrt2) (Monte Carlo, n=4e5)", abs(auc_mc - auc_th), tol=3e-3)

# E2. Smallest Mahalanobis move that sets w.Delta' = target is along Sigma w (Lagrange, symbolic).
# minimise (D'-D)' S^-1 (D'-D) s.t. w.D' = c  ->  D' = D + (c - w.D)/(w'Sw) * S w
dd = 3
Ssym = sp.Matrix(dd, dd, lambda i, j: sp.Symbol(f"s{min(i,j)}{max(i,j)}"))
D = sp.Matrix(sp.symbols("D0:3"))
W = sp.Matrix(sp.symbols("w0:3"))
X = sp.Matrix(sp.symbols("x0:3"))
c, lmb = sp.symbols("c lmb")
Sinv = Ssym.inv()
lag = ((X - D).T * Sinv * (X - D))[0] - lmb * ((W.T * X)[0] - c)
# stationarity: 2 S^-1 (X - D) = lmb W  ->  X = D + lmb/2 S W ; plug into constraint
X_star = D + (c - (W.T * D)[0]) / (W.T * Ssym * W)[0] * Ssym * W
grad = [sp.diff(lag, xi) for xi in X]
lmb_star = 2 * (c - (W.T * D)[0]) / (W.T * Ssym * W)[0]
res = [sp.simplify(g.subs(dict(zip(X, X_star))).subs(lmb, lmb_star)) for g in grad]
cons = sp.simplify((W.T * X_star)[0] - c)
check("E2 min Mahalanobis move is along Sigma w (symbolic KKT)",
      0.0 if all(r == 0 for r in res) and cons == 0 else 1.0)
# numeric: compare with brute-force minimiser over random feasible directions
w = rng.normal(size=d)
target = 0.0
move = (target - w @ delta) / (w @ S @ w) * S @ w
Si = np.linalg.inv(S)
cost_star = move @ Si @ move
worst = np.inf
for _ in range(20000):
    v = rng.normal(size=d)
    if abs(w @ v) < 1e-3:
        continue
    m = (target - w @ delta) / (w @ v) * v
    worst = min(worst, m @ Si @ m)
check("E2b no random feasible move is cheaper", max(0.0, cost_star - worst))
# and the Euclidean minimiser is along w instead
move_e = (target - w @ delta) / (w @ w) * w
check("E2c Euclidean min move along w is Mahalanobis-costlier",
      0.0 if move_e @ Si @ move_e >= cost_star - 1e-12 else 1.0)

# E3. Core lemma: after Delta' = Delta - t S w (S fixed), d'_u' = d'_u - k d'_w cos_S(u,w),
# k = fraction of w's separation removed. Symbolic: every quantity is a bilinear form, so write
# u.D = uD, w.D = wD, u'Sw = uSw, u'Su = qU > 0, w'Sw = qW > 0; then u.D' = uD - t uSw and
# w.D' = wD - t qW (exact expansion of D' = D - t S w).
t, uD, wD, uSw = sp.symbols("t uD wD uSw", real=True)
qU, qW = sp.symbols("qU qW", positive=True)
dpu_before, dpu_after = uD / sp.sqrt(qU), (uD - t * uSw) / sp.sqrt(qU)
dpw_before, dpw_after = wD / sp.sqrt(qW), (wD - t * qW) / sp.sqrt(qW)
k_expr = 1 - dpw_after / dpw_before
cosS = uSw / sp.sqrt(qU * qW)
diff = sp.simplify(dpu_after - (dpu_before - k_expr * dpw_before * cosS))
check("E3 d'_u(after) = d'_u - k d'_w cos_Sigma (symbolic)", 0.0 if diff == 0 else 1.0)
# numeric spot check across many random u, partial and full removal
errs = []
for kk in (0.3, 1.0):
    tt = kk * (w @ delta) / (w @ S @ w)
    dnew = delta - tt * S @ w
    for _ in range(200):
        uu = rng.normal(size=d)
        errs.append(abs(dprime(uu, dnew, S)
                        - (dprime(uu, delta, S) - kk * dprime(w, delta, S) * cos_S(uu, w, S))))
check("E3b numeric (400 probes, k in {0.3,1})", max(errs))

# E4. cos_Sigma(u,w) = Pearson correlation of the two probes' scores within a class.
xs = rng.normal(size=(n, d)) @ L.T
r = np.corrcoef(xs @ u, xs @ w)[0, 1]
check("E4 cos_Sigma = within-class correlation of probe scores (MC)", abs(r - cos_S(u, w, S)), tol=3e-3)

# E5. Refit: best linear separation after the move, d'_opt^2 = Delta' S^-1 Delta'
#      = Delta' S^-1 Delta - k(2-k) d'_w^2 ; zero iff w ∝ S^-1 Delta and k = 1.
errs = []
for kk in (0.3, 0.7, 1.0):
    tt = kk * (w @ delta) / (w @ S @ w)
    dnew = delta - tt * S @ w
    lhs = dnew @ Si @ dnew
    rhs = delta @ Si @ delta - kk * (2 - kk) * dprime(w, delta, S) ** 2
    errs.append(abs(lhs - rhs))
w_opt = Si @ delta
dnew = delta - (w_opt @ delta) / (w_opt @ S @ w_opt) * S @ w_opt
errs.append(np.linalg.norm(dnew))
check("E5 refit d'^2 = D_opt^2 - k(2-k) d'_w^2; collapses for optimal w", max(errs))

# E6. Diagnostic (projection-collapse evader): f(x) = x - S w (w'x - b)/(w'Sw) at full removal.
#      Probe u then acts as u_eff = u - (u'Sw / w'Sw) w on clean data:
#      d'_u(after) = (d'_u - c d'_w) / sqrt(1 - c^2).
errs = []
for _ in range(200):
    uu = rng.normal(size=d)
    ueff = uu - (uu @ S @ w) / (w @ S @ w) * w
    cc = cos_S(uu, w, S)
    errs.append(abs(dprime(ueff, delta, S)
                    - (dprime(uu, delta, S) - cc * dprime(w, delta, S)) / np.sqrt(1 - cc ** 2)))
check("E6 projection-collapse variant (200 probes)", max(errs))

# E7. Euclidean-cost evader: Delta' = Delta - t w (the smallest Euclidean move). Then
#      d'_u(after) = d'_u - k d'_w cos(u,w) |u| sqrt(w'Sw) / (|w| sqrt(u'Su)), cos = plain cosine.
uw, nU, nW = sp.symbols("uw nU nW", positive=True)  # u.w, |u|, |w|
dpw_after_e = (wD - t * nW**2) / sp.sqrt(qW)
k_e = 1 - dpw_after_e / dpw_before
lhs_e = (uD - t * uw) / sp.sqrt(qU)
rhs_e = dpu_before - k_e * dpw_before * (uw / (nU * nW)) * nU * sp.sqrt(qW) / (nW * sp.sqrt(qU))
check("E7 Euclidean-move predictor (symbolic)", 0.0 if sp.simplify(lhs_e - rhs_e) == 0 else 1.0)
errs = []
for kk in (0.4, 1.0):
    dnew = delta - kk * (w @ delta) / (w @ w) * w
    for _ in range(200):
        uu = rng.normal(size=d)
        cos = uu @ w / (np.linalg.norm(uu) * np.linalg.norm(w))
        pred = dprime(uu, delta, S) - kk * dprime(w, delta, S) * cos * np.linalg.norm(uu) * np.sqrt(w @ S @ w) / (
            np.linalg.norm(w) * np.sqrt(uu @ S @ uu))
        errs.append(abs(dprime(uu, dnew, S) - pred))
check("E7b numeric (400 probes, k in {0.4,1})", max(errs))

print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
