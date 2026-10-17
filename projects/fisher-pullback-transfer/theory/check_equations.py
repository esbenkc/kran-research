"""Symbolic and numeric checks of every equation in theory/lemma.md (and the post).

Run: uv run python projects/fisher-pullback-transfer/theory/check_equations.py
Writes theory/check_equations.out. Pure numpy/sympy/scipy, seed 0, < 10 s on CPU.
Each check prints PASS/FAIL and the number it compared.
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


# ---------------------------------------------------------------------------
# E1. Second-order retain cost: KL(p(h) || p(h+d)) = 1/2 d' F d + O(|d|^3),
#     F = J' (diag p - p p') J, for logits z = A tanh(B h) (a nonlinear map to V outputs).
d, V, H = 12, 30, 20
A = rng.normal(size=(V, H)) / np.sqrt(H)
B = rng.normal(size=(H, d)) / np.sqrt(d)
h0 = rng.normal(size=d)


def logits(h):
    return A @ np.tanh(B @ h)


def jac(h):
    return A @ np.diag(1 - np.tanh(B @ h) ** 2) @ B  # V x d


p0 = softmax(logits(h0))
J = jac(h0)
F_out = np.diag(p0) - np.outer(p0, p0)
F = J.T @ F_out @ J
dirn = rng.normal(size=d)
dirn /= np.linalg.norm(dirn)
ratios = []
for eps in [1e-1, 1e-2, 1e-3]:
    dd = eps * dirn
    ratios.append(kl(p0, softmax(logits(h0 + dd))) / (0.5 * dd @ F @ dd))
report("E1 KL ~ 1/2 d'Fd (ratio -> 1 as eps -> 0)", abs(ratios[-1] - 1) < 1e-2,
       f"ratios at eps=1e-1,1e-2,1e-3: {[round(float(r), 5) for r in ratios]}")

# E1b. True Fisher by sampling labels from the model equals J'(diag p - pp')J:
#      E_{y~p}[g g'], g = d/dh log p(y|h) = J'(e_y - p). Exact expectation over y.
g_all = J.T @ (np.eye(V) - p0[None, :]).T  # d x V, column y = J'(e_y - p)
F_sample_exact = (g_all * p0[None, :]) @ g_all.T
report("E1b E_y~p[g g'] = J'(diag p - pp')J", np.allclose(F_sample_exact, F, atol=1e-12),
       f"max abs diff {np.abs(F_sample_exact - F).max():.2e}")

# E1c. Empirical Fisher (data labels, not model samples) is a DIFFERENT matrix: with labels from
#      another distribution q, E_{y~q}[g g'] != F.
q = softmax(rng.normal(size=V))
F_emp = (g_all * q[None, :]) @ g_all.T
report("E1c empirical Fisher differs from true Fisher", np.abs(F_emp - F).max() > 1e-3,
       f"max abs diff {np.abs(F_emp - F).max():.3e}")

# ---------------------------------------------------------------------------
# E2. Common-offset Fisher for a sequence: outputs at position s depend on h_1..h_s.
#     Adding the SAME offset d at every position costs 1/2 d'(sum_{t,t'} F_tt')d = 1/2 d' E[g g'] d,
#     with g = sum_t d/dh_t sum_s log p(y_s), y_s ~ p. Block-diagonal sum_t F_tt misses cross terms.
T = 3
Ws = [[rng.normal(size=(V, d)) / np.sqrt(d) for _ in range(T)] for _ in range(T)]  # Ws[s][t]
hs = [rng.normal(size=d) for _ in range(T)]


def seq_logits(hs_):
    return [sum(Ws[s][t] @ np.tanh(hs_[t]) for t in range(s + 1)) for s in range(T)]


def seq_jac(s, t):  # d z_s / d h_t
    return Ws[s][t] @ np.diag(1 - np.tanh(hs[t]) ** 2) if t <= s else np.zeros((V, d))


ps = [softmax(z) for z in seq_logits(hs)]
F_full = np.zeros((d, d))   # sum_{t,t'} F_tt' = sum_s Jbar_s' Fout_s Jbar_s, Jbar_s = sum_t dz_s/dh_t
F_block = np.zeros((d, d))  # sum_t F_tt
for s in range(T):
    Fo = np.diag(ps[s]) - np.outer(ps[s], ps[s])
    Jbar = sum(seq_jac(s, t) for t in range(T))
    F_full += Jbar.T @ Fo @ Jbar
    for t in range(T):
        F_block += seq_jac(s, t).T @ Fo @ seq_jac(s, t)
# Monte Carlo of E[g g'] with sampled y (the estimator the experiment uses)
n_mc = 200_000
G = np.zeros((n_mc, d))
for s in range(T):
    Jbar = sum(seq_jac(s, t) for t in range(T))
    ys = rng.choice(V, size=n_mc, p=ps[s])
    E = -np.tile(ps[s], (n_mc, 1))
    E[np.arange(n_mc), ys] += 1
    G += E @ Jbar  # rows: g' contributions; independent across s given h
F_mc = G.T @ G / n_mc
rel_mc = np.linalg.norm(F_mc - F_full) / np.linalg.norm(F_full)
eps = 1e-3
kl_tot = sum(kl(ps[s], softmax(z)) for s, z in enumerate(seq_logits([h + eps * dirn for h in hs])))
ratio_full = kl_tot / (0.5 * eps ** 2 * dirn @ F_full @ dirn)
ratio_block = kl_tot / (0.5 * eps ** 2 * dirn @ F_block @ dirn)
report("E2 common-offset KL = 1/2 d'(sum_tt' F_tt')d", abs(ratio_full - 1) < 1e-2,
       f"ratio full {ratio_full:.5f}, ratio block-diagonal {ratio_block:.5f} (should differ from 1)")
report("E2b sampled-label estimator E[g g'] -> sum_tt' F_tt'", rel_mc < 1e-2,
       f"relative Frobenius error {rel_mc:.2e} at n={n_mc}")

# ---------------------------------------------------------------------------
# E3. Cheapest move against probe w: min 1/2 d'F d s.t. w'd = -eps  =>  d* = -eps F^-1 w / (w'F^-1 w).
#     Symbolic (Lagrange) in 3-D, then numeric in d=12 against a generic solver.
x1, x2, x3, lam, ep = sp.symbols("x1 x2 x3 lam ep", real=True)
Fs = sp.Matrix([[3, 1, 0], [1, 2, sp.Rational(1, 2)], [0, sp.Rational(1, 2), 1]])
ws = sp.Matrix([1, -2, 3])
xs = sp.Matrix([x1, x2, x3])
Lg = sp.Rational(1, 2) * (xs.T * Fs * xs)[0] + lam * ((ws.T * xs)[0] + ep)
sol = sp.solve([sp.diff(Lg, v) for v in (x1, x2, x3, lam)], (x1, x2, x3, lam), dict=True)[0]
d_sym = sp.Matrix([sol[x1], sol[x2], sol[x3]])
d_claim = -ep * Fs.inv() * ws / (ws.T * Fs.inv() * ws)[0]
report("E3 Lagrange optimum = -eps F^-1 w / w'F^-1 w (sympy)",
       sp.simplify(d_sym - d_claim) == sp.zeros(3, 1), f"d* = {list(sp.simplify(d_sym))}")
Fn = F + 1e-3 * np.eye(d)
w = rng.normal(size=d)
u = rng.normal(size=d)
dstar = -1.0 * np.linalg.solve(Fn, w) / (w @ np.linalg.solve(Fn, w))
# any other feasible move costs more
others = []
for _ in range(1000):
    z = rng.normal(size=d)
    z = z - (w @ z + 1.0) / (w @ w) * w  # project to w'z = -1
    others.append(0.5 * z @ Fn @ z)
report("E3b no random feasible move is cheaper", min(others) >= 0.5 * dstar @ Fn @ dstar - 1e-12,
       f"cost(d*) {0.5 * dstar @ Fn @ dstar:.4e} <= min random {min(others):.4e}")

# E4. Transfer of the cheapest move to probe u: u'd* = -eps * r_P(u,w), r_P = u'Pw / w'Pw, P = F^-1.
P = np.linalg.inv(Fn)
r = (u @ P @ w) / (w @ P @ w)
report("E4 u'd* = -eps r_P(u,w)", np.isclose(u @ dstar, -1.0 * r), f"u'd* {u @ dstar:.6f}, -r {-r:.6f}")
# r_P is invariant to rescaling u's norm only up to the factor (it scales with u), and invariant to the
# scale of P (damping-free part): check P -> 7P leaves r unchanged.
report("E4b r_P invariant to P -> c P", np.isclose(r, (u @ (7 * P) @ w) / (w @ (7 * P) @ w)), "c = 7")

# ---------------------------------------------------------------------------
# E5. Edit model and calibration. dh(x) = -(a + b s_w(x)) P w.
#     Target gap change: delta_w = -b (w'Pw) G_w(w).  Then for any probe u:
#     s_u_new(x) = s_u(x) - a u'Pw + (delta_w / G_w(w)) r_P(u,w) s_w(x).
a_, b_, sw, su, wPw, uPw, Gw = sp.symbols("a b s_w s_u wPw uPw G_w", real=True)
delta_w = -b_ * wPw * Gw
su_new = su - (a_ + b_ * sw) * uPw
su_claim = su - a_ * uPw + (delta_w / Gw) * (uPw / wPw) * sw
report("E5 s_u_new = s_u - a u'Pw + (delta_w/G_w) r_P s_w (sympy)", sp.simplify(su_new - su_claim) == 0,
       "identity holds")
# Numeric: Gaussian data for two different concepts; the constant a leaves AUROC unchanged, and the
# predicted AUROC from the formula equals the AUROC of the actually edited activations.
n = 4000
X = rng.normal(size=(n, d)) @ np.linalg.cholesky(np.cov(rng.normal(size=(50, d)).T) + 0.1 * np.eye(d)).T
yw = (X @ rng.normal(size=d) + rng.normal(size=n) > 0).astype(int)  # target concept labels
yu = (X @ rng.normal(size=d) + rng.normal(size=n) > 0).astype(int)  # another concept, same inputs
s_w = X @ w
G_w = s_w[yw == 1].mean() - s_w[yw == 0].mean()
a_n, b_n = 0.7, 0.3 / (w @ P @ w)
X_new = X - (a_n + b_n * s_w)[:, None] * (P @ w)[None, :]
d_w = (X_new @ w)[yw == 1].mean() - (X_new @ w)[yw == 0].mean() - G_w  # measured target gap change
s_u = X @ u
pred = s_u + (d_w / G_w) * r * s_w
auc_true = roc_auc_score(yu, X_new @ u)
auc_pred = roc_auc_score(yu, pred)
report("E5b predicted AUROC = AUROC of edited activations (a drops out)", abs(auc_true - auc_pred) < 1e-12,
       f"{auc_true:.6f} vs {auc_pred:.6f}; clean {roc_auc_score(yu, s_u):.4f}")

# E6. On the target's own data the affine-in-s_w edit only rescales s_w: its AUROC is unchanged unless
#     b w'Pw > 1 (a flip). So delta_w calibrates b; it does not by itself explain the target's own drop.
report("E6 target AUROC unchanged by an affine-in-s_w edit (b w'Pw < 1)",
       abs(roc_auc_score(yw, X_new @ w) - roc_auc_score(yw, s_w)) < 1e-12,
       f"{roc_auc_score(yw, X_new @ w):.6f} vs {roc_auc_score(yw, s_w):.6f}")

# ---------------------------------------------------------------------------
# E7. Reduction to the 2026-10-16 law: same labels for u and w (y_u = y_w), P = Sigma (whitened):
#     gap change of u divided by sd_u equals k d'_w cos_Sigma(u,w), k = -delta_w / G_w.
Sig = np.cov(X.T)
Pw_s = Sig
r_s = (u @ Pw_s @ w) / (w @ Pw_s @ w)
Gu_w = G_w  # same labels => w-gap on u's data = target gap
k = 0.4
dgap_u = (-k * G_w) / G_w * r_s * Gu_w
lhs = dgap_u / np.sqrt(u @ Sig @ u)
dprime_w = G_w / np.sqrt(w @ Sig @ w)
cos_s = (u @ Sig @ w) / np.sqrt((u @ Sig @ u) * (w @ Sig @ w))
report("E7 same-label case reduces to -k d'_w cos_Sigma(u,w)", np.isclose(lhs, -k * dprime_w * cos_s),
       f"{lhs:.6f} vs {-k * dprime_w * cos_s:.6f}")

# E8. Literal cosine equals the ratio form only when u'P u is proportional to u'Sigma u across probes:
#     r_P / sqrt(u'Su) * sqrt(w'Sw) = cos_P(u,w) * sqrt(u'Pu/u'Su) / sqrt(w'Pw/w'Sw).
cosP = (u @ P @ w) / np.sqrt((u @ P @ u) * (w @ P @ w))
lhs8 = r * np.sqrt(w @ Sig @ w) / np.sqrt(u @ Sig @ u)
rhs8 = cosP * np.sqrt((u @ P @ u) / (u @ Sig @ u)) / np.sqrt((w @ P @ w) / (w @ Sig @ w))
report("E8 ratio form vs literal cosine identity", np.isclose(lhs8, rhs8), f"{lhs8:.6f} vs {rhs8:.6f}")

# E9. Equal-variance Gaussian classes: AUROC = Phi(d'/sqrt 2).
dp = 1.3
z0 = rng.normal(size=200_000)
z1 = rng.normal(size=200_000) + dp
auc = roc_auc_score(np.r_[np.zeros_like(z0), np.ones_like(z1)], np.r_[z0, z1])
report("E9 AUROC = Phi(d'/sqrt2)", abs(auc - norm.cdf(dp / np.sqrt(2))) < 3e-3,
       f"empirical {auc:.4f}, formula {norm.cdf(dp / np.sqrt(2)):.4f}")

# E10. Damping rule lambda = c * median eig(F) is scale-equivariant: F -> sF gives the same (F+lam I)^-1 w
#      direction and the same r_P.
Fr = F_full
lam_ = np.median(np.linalg.eigvalsh(Fr))
r1 = (u @ np.linalg.solve(Fr + lam_ * np.eye(d), w)) / (w @ np.linalg.solve(Fr + lam_ * np.eye(d), w))
s = 37.0
lam2 = np.median(np.linalg.eigvalsh(s * Fr))
r2 = (u @ np.linalg.solve(s * Fr + lam2 * np.eye(d), w)) / (w @ np.linalg.solve(s * Fr + lam2 * np.eye(d), w))
report("E10 median-eigenvalue damping is scale-equivariant", np.isclose(r1, r2), f"{r1:.6f} vs {r2:.6f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
