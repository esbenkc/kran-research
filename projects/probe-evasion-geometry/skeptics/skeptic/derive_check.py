"""Skeptic's own derivation check (independent of build's code).

Model: x | y ~ N(mu_y, S), y in {0,1}, dmu = mu1 - mu0.
Linear probe u: score u.x. Class-conditional scores are Gaussian with equal variance,
so AUROC(u) = Phi(d_u / sqrt 2), d_u = u.dmu / sqrt(u' S u).

Claim A: smallest move of the class-1 mean that sets w.dmu' = 0
  - in Mahalanobis norm (delta' S^-1 delta) -> delta = -(w.dmu / w'Sw) S w
  - in Euclidean norm                      -> delta = -(w.dmu / w'w) w
Claim B: after the Mahalanobis move, d_u' = d_u - sqrt(w'Sw)*... * cos_S(w,u) * d_w
  with cos_S(w,u) = u'Sw / sqrt(u'Su w'Sw).  (linear in whitened cosine, slope d_w)
Claim C: refit recovers iff dmu' != 0; dmu' = 0 iff dmu parallel to S w,
  i.e. iff w is the LDA direction S^-1 dmu.
"""
import numpy as np
import sympy as sp
from scipy.stats import norm

# --- Claim A (sympy, Lagrange, symbolic in w, dmu, lambda; random rational SPD Sigma) ---
d = sp.Matrix(sp.symbols("d0:3"))
w = sp.Matrix(sp.symbols("w0:3"))
dm = sp.Matrix(sp.symbols("m0:3"))
lam = sp.symbols("lam")
A = sp.Matrix([[2, 1, 0], [sp.Rational(1, 3), 1, 1], [0, -1, 3]])
S = A * A.T + sp.eye(3)  # SPD
Sinv = S.inv()
L = (d.T * Sinv * d)[0] - lam * ((w.T * (dm + d))[0])
eqs = list(sp.Matrix([L]).jacobian(d)) + [(w.T * (dm + d))[0]]
sol = sp.solve(eqs, list(d) + [lam], dict=True)[0]
dsol = sp.Matrix([sol[v] for v in d])
pred = -((w.T * dm)[0] / (w.T * S * w)[0]) * S * w
print("Claim A (Mahalanobis min-move = -(w.dmu/w'Sw) S w), symbolic in w,dmu:",
      all(sp.simplify(x) == 0 for x in (dsol - pred)))

# --- Claim B, C numeric ---
def auroc(u, dmu, S):
    return norm.cdf((u @ dmu) / np.sqrt(u @ S @ u) / np.sqrt(2))

D = 16
rng = np.random.default_rng(1)
Q = rng.normal(size=(D, D)); S = Q @ Q.T / D + 0.1 * np.eye(D)
dmu = rng.normal(size=D)
w = rng.normal(size=D)
dw = (w @ dmu) / np.sqrt(w @ S @ w)
delta = -((w @ dmu) / (w @ S @ w)) * S @ w
dmu2 = dmu + delta
ok = True
for _ in range(200):
    u = rng.normal(size=D)
    du = (u @ dmu) / np.sqrt(u @ S @ u)
    du2 = (u @ dmu2) / np.sqrt(u @ S @ u)
    cosS = (u @ S @ w) / np.sqrt((u @ S @ u) * (w @ S @ w))
    ok &= np.isclose(du - du2, dw * cosS)
print("Claim B (d_u drop = d_w * cos_S(w,u)):", ok)
print("  probe w after move AUROC:", auroc(w, dmu2, S))
lda = np.linalg.solve(S, dmu2)
print("  refit LDA AUROC after move (w random):", auroc(lda, dmu2, S))
wl = np.linalg.solve(S, dmu)  # w = LDA direction
dl = -((wl @ dmu) / (wl @ S @ wl)) * S @ wl
print("Claim C: |dmu'| when w is the LDA probe:", np.linalg.norm(dmu + dl), "(collapse)")

# Euclidean min-move gives a different predictor: u.w / sqrt(u'Su), not cos_S
dE = -((w @ dmu) / (w @ w)) * w
u = rng.normal(size=D)
print("Euclidean drop", (u @ -dE) / np.sqrt(u @ S @ u),
      " vs dw*cosS", dw * (u @ S @ w) / np.sqrt((u @ S @ u) * (w @ S @ w)), "(differ in general)")
