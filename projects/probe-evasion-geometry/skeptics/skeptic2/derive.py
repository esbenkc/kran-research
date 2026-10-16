"""skeptic2: independent derivation checks for the probe-evasion lemma.

Model: x | y ~ N(mu_y, Sigma), y in {0,1}, delta = mu_1 - mu_0.
Probe v scores s = v^T x.  a_v := v^T delta / sqrt(2 v^T Sigma v);  AUROC(v) = Phi(a_v).

Checks (each prints PASS/FAIL):
 C1 binormal AUROC formula (Monte Carlo)
 C2 min Sigma^{-1}-norm move that nulls probe w is d = -(w^T delta / w^T Sigma w) Sigma w   (sympy, KKT)
 C3 min Euclidean-norm move is d = -(w^T delta / w^T w) w   (sympy, KKT)
 C4 after the Sigma-move: a_u' = a_u - a_w cos_Sigma(u,w)   (sympy identity, general n via random numeric + 2D symbolic)
 C5 refit after the Sigma-move: best separation a*' = a* sin(theta),  cos(theta) = a_w / a*;
    so attacking the optimal probe w = Sigma^{-1} delta makes delta' = 0 (full collapse)
 C6 partial evasion (move a fraction t): a_u' = a_u - t a_w cos_Sigma(u,w)
 C7 linear encoder + Euclidean reconstruction + probe penalty: optimum moves h along w, not Sigma w
"""
import numpy as np
import sympy as sp
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
ok = lambda b: "PASS" if b else "FAIL"


def a(v, delta, S):
    return v @ delta / np.sqrt(2 * v @ S @ v)


def cosS(u, w, S):
    return u @ S @ w / np.sqrt((u @ S @ u) * (w @ S @ w))


# ---------- C1 ----------
n = 6
L = rng.normal(size=(n, n)); S = L @ L.T + 0.3 * np.eye(n)
delta = rng.normal(size=n)
u = rng.normal(size=n)
N = 400_000
X0 = rng.multivariate_normal(np.zeros(n), S, N)
X1 = rng.multivariate_normal(delta, S, N)
auc_mc = roc_auc_score(np.r_[np.zeros(N), np.ones(N)], np.r_[X0 @ u, X1 @ u])
auc_th = norm.cdf(a(u, delta, S))
print(f"C1 binormal AUROC: MC={auc_mc:.4f} theory={auc_th:.4f}  {ok(abs(auc_mc-auc_th)<3e-3)}")

# ---------- C2 / C3 (sympy, 3D, fully symbolic Sigma) ----------
k = 3
Ssym = sp.Matrix(k, k, lambda i, j: sp.Symbol(f"s{min(i,j)}{max(i,j)}"))
wv = sp.Matrix(sp.symbols("w0:3")); dl = sp.Matrix(sp.symbols("d0:3"))
lam = sp.Symbol("lam")
# candidate Sigma-move
c = (wv.T * dl)[0] / (wv.T * Ssym * wv)[0]
dS = -c * Ssym * wv
# KKT for min d^T Sigma^{-1} d s.t. w^T(delta+d)=0 : Sigma^{-1} d = lam w  <=>  d = lam Sigma w
feas = sp.simplify((wv.T * (dl + dS))[0])
print(f"C2 Sigma-move feasible (w^T delta'=0): {ok(feas == 0)} ; stationarity d ∝ Sigma w holds by construction (lam={sp.simplify(-c)})")
# convexity: objective strictly convex for Sigma PD and constraint affine -> unique minimiser; numeric check vs random feasible points
w_ = rng.normal(size=n)
dS_ = -(w_ @ delta) / (w_ @ S @ w_) * S @ w_
Si = np.linalg.inv(S)
best = dS_ @ Si @ dS_
worse = []
for _ in range(2000):
    z = rng.normal(size=n); z -= (w_ @ z) / (w_ @ w_) * w_  # null-space perturbation
    d = dS_ + z * rng.uniform(0, 1)
    worse.append(d @ Si @ d >= best - 1e-12)
print(f"C2 Sigma-move is min Sigma^-1 norm over 2000 feasible perturbations: {ok(all(worse))}")
dE_ = -(w_ @ delta) / (w_ @ w_) * w_
worse = []
for _ in range(2000):
    z = rng.normal(size=n); z -= (w_ @ z) / (w_ @ w_) * w_
    d = dE_ + z * rng.uniform(0, 1)
    worse.append(d @ d >= dE_ @ dE_ - 1e-12)
print(f"C3 Euclid-move is min L2 norm: {ok(all(worse))};  Euclid-move != Sigma-move unless w is a Sigma eigvec: angle(dE,dS)={np.degrees(np.arccos(abs(dE_@dS_)/np.linalg.norm(dE_)/np.linalg.norm(dS_))):.1f} deg")

# ---------- C4 (symbolic identity, 3D) ----------
uv = sp.Matrix(sp.symbols("u0:3"))
aS = lambda v, d: (v.T * d)[0] / sp.sqrt(2 * (v.T * Ssym * v)[0])
cos_sym = (uv.T * Ssym * wv)[0] / sp.sqrt((uv.T * Ssym * uv)[0] * (wv.T * Ssym * wv)[0])
# use positive symbols for the quadratic forms so sqrt(Qu*Qw)=sqrt(Qu)sqrt(Qw) is valid
Qu, Qw = sp.symbols("Qu Qw", positive=True)
ud, wd, uSw = sp.symbols("ud wd uSw", real=True)
# a_u' = u^T(delta - (wd/Qw) Sigma w)/sqrt(2Qu) = (ud - wd*uSw/Qw)/sqrt(2Qu)
lhs = (ud - wd * uSw / Qw) / sp.sqrt(2 * Qu)
rhs = ud / sp.sqrt(2 * Qu) - wd / sp.sqrt(2 * Qw) * uSw / sp.sqrt(Qu * Qw)
# check the reduction u^T dS = -(wd/Qw) uSw symbolically with full 3x3 Sigma
red = sp.simplify((uv.T * dS)[0] + (wv.T * dl)[0] * (uv.T * Ssym * wv)[0] / (wv.T * Ssym * wv)[0])
diff = sp.simplify(lhs - rhs)
print(f"C4 a_u' = a_u - a_w cos_Sigma(u,w): reduction {ok(red == 0)}, identity {ok(diff == 0)}")
# numeric in n=50
n2 = 50
L2 = rng.normal(size=(n2, n2)); S2 = L2 @ L2.T / n2 + 0.05 * np.eye(n2)
d2 = 0.25 * rng.normal(size=n2); w2 = rng.normal(size=n2)
d2p = d2 - (w2 @ d2) / (w2 @ S2 @ w2) * S2 @ w2
errs = [abs(a(uu, d2p, S2) - (a(uu, d2, S2) - a(w2, d2, S2) * cosS(uu, w2, S2))) for uu in rng.normal(size=(200, n2))]
print(f"C4 numeric n=50, 200 probes: max err {max(errs):.2e}  {ok(max(errs)<1e-9)}")

# ---------- C5 refit ----------
astar = lambda d, S: np.sqrt(d @ np.linalg.solve(S, d) / 2)
cos_th = a(w2, d2, S2) / astar(d2, S2)
print(f"C5 refit a*' = a* sin(theta): {astar(d2p,S2):.6f} vs {astar(d2,S2)*np.sqrt(1-cos_th**2):.6f}  {ok(abs(astar(d2p,S2)-astar(d2,S2)*np.sqrt(1-cos_th**2))<1e-9)}")
wopt = np.linalg.solve(S2, d2)
d2c = d2 - (wopt @ d2) / (wopt @ S2 @ wopt) * S2 @ wopt
print(f"C5 attack optimal probe -> |delta'| = {np.linalg.norm(d2c):.2e} (collapse, refit AUROC = 0.5)  {ok(np.linalg.norm(d2c)<1e-9)}")
# a trained logistic probe is ~ optimal: refit AUROC after its min-Sigma move
for frac_noise in [0.0, 0.1, 0.3]:
    wn = wopt + frac_noise * np.linalg.norm(wopt) * rng.normal(size=n2) / np.sqrt(n2)
    dd = d2 - (wn @ d2) / (wn @ S2 @ wn) * S2 @ wn
    print(f"    attacked probe = optimal + {frac_noise:.0%} noise: AUROC(w) before {norm.cdf(a(wn,d2,S2)):.3f}, refit AUROC after {norm.cdf(astar(dd,S2)):.3f} (orig best {norm.cdf(astar(d2,S2)):.3f})")

# ---------- C6 partial ----------
t = 0.4
d2t = d2 - t * (w2 @ d2) / (w2 @ S2 @ w2) * S2 @ w2
errs = [abs(a(uu, d2t, S2) - (a(uu, d2, S2) - t * a(w2, d2, S2) * cosS(uu, w2, S2))) for uu in rng.normal(size=(200, n2))]
print(f"C6 partial move t={t}: max err {max(errs):.2e}  {ok(max(errs)<1e-9)}")

# ---------- C7 linear encoder, Euclidean reconstruction ----------
# min_B E||B x||^2 + lam (w^T (I+B) delta)^2 ; stationarity: B M + lam r w delta^T = 0 -> B = -lam r w delta^T M^{-1}
# => every representation change Bx is parallel to w (rank-1, column space = span(w)), not Sigma w.
p = 0.5
mu0 = rng.normal(size=n); mu1 = mu0 + delta
M = S + p * np.outer(mu1, mu1) + (1 - p) * np.outer(mu0, mu0)
lamv = 50.0
# solve the 1-D problem for scalar c in B = -c w delta^T M^{-1}
from scipy.optimize import minimize_scalar
Mi = np.linalg.inv(M)
def obj(cc):
    B = -cc * np.outer(w_, delta @ Mi)
    return np.trace(B @ M @ B.T) + lamv * (w_ @ (np.eye(n) + B) @ delta) ** 2
cc = minimize_scalar(obj).x
B = -cc * np.outer(w_, delta @ Mi)
# verify it is the global optimum of the full matrix problem via gradient norm
G = 2 * B @ M + 2 * lamv * (w_ @ (np.eye(n) + B) @ delta) * np.outer(w_, delta)
print(f"C7 linear enc + L2 recon: grad norm at rank-1 B along w = {np.linalg.norm(G):.2e} {ok(np.linalg.norm(G)<1e-6)}")
dmean = B @ delta
print(f"    mean-diff move direction vs w: |cos|={abs(dmean@w_)/np.linalg.norm(dmean)/np.linalg.norm(w_):.4f}; vs Sigma w: |cos|={abs(dmean@S@w_)/np.linalg.norm(dmean)/np.linalg.norm(S@w_):.4f}")
# under KL / Sigma^{-1} preservation cost the same algebra gives the Sigma-move:
#   min_d 1/2 d^T Sigma^{-1} d  (= KL(N(mu+d,Sigma)||N(mu,Sigma)))  s.t. w^T(delta+d)=0  -> C2.
print("    => lemma's 'along Sigma w' is an ASSUMPTION about the evader's cost metric (Sigma^{-1}/KL), not a consequence of a probe penalty alone.")

# ---------- C8 linear encoder, Mahalanobis retain, centred data: w-score variance -> w'Sigma w (1 - cos^2 theta) ----------
# B = Sigma w b', b = alpha Sigma^-1 delta, alpha = -(w'delta)/(w'Sigma w * D^2)  (min E||Sigma^-1/2 B x||^2 s.t. w-gap = 0)
Dsq = delta @ np.linalg.solve(S, delta)
alpha = -(w_ @ delta) / ((w_ @ S @ w_) * Dsq)
Bm = np.outer(S @ w_, alpha * np.linalg.solve(S, delta))
Ip = np.eye(n) + Bm
gap_after = w_ @ Ip @ delta
var_after = w_ @ Ip @ S @ Ip.T @ w_
cos2 = (w_ @ delta) ** 2 / (w_ @ S @ w_) / Dsq
print(f"C8 Mahalanobis linear encoder: w-gap after {gap_after:.1e}; var ratio {var_after/(w_@S@w_):.6f} vs 1-cos^2theta {1-cos2:.6f}  {ok(abs(gap_after)<1e-10 and abs(var_after/(w_@S@w_)-(1-cos2))<1e-10)}")
# optimality of b in the centred case (M = Sigma + p(1-p) delta delta'): grad of b'Mb on constraint surface ∝ delta
M_c = S + 0.25 * np.outer(delta, delta)
bvec = alpha * np.linalg.solve(S, delta)
g = M_c @ bvec
print(f"C8 b ∝ M^-1 delta (Sherman-Morrison): |cos(M b, delta)| = {abs(g@delta)/np.linalg.norm(g)/np.linalg.norm(delta):.6f}  {ok(abs(abs(g@delta)/np.linalg.norm(g)/np.linalg.norm(delta)-1)<1e-10)}")
