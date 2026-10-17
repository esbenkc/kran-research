"""Vera's numeric checks of derivation.md (independent of build code). numpy only."""
import numpy as np

rng = np.random.default_rng(0)
d, V = 8, 12
W = rng.normal(size=(V, d))          # logits = W h  (J = W)
h = rng.normal(size=d)


def logp(h):
    z = W @ h
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


# 1. KL(p(h) || p(h+e)) ≈ 1/2 e' J' F_out J e
p = np.exp(logp(h))
Fout = np.diag(p) - np.outer(p, p)
F = W.T @ Fout @ W
for eps in [1e-1, 1e-2, 1e-3]:
    e = eps * rng.normal(size=d)
    kl = (p * (logp(h) - logp(h + e))).sum()
    quad = 0.5 * e @ F @ e
    print(f"[1] eps={eps:g}  KL={kl:.3e}  quad={quad:.3e}  rel.err={abs(kl-quad)/quad:.2e}")

# true Fisher = E_{y~p}[g g'], g = d log p(y)/dh = W'(e_y - p)
Fs = sum(p[y] * np.outer(W.T @ (np.eye(V)[y] - p), W.T @ (np.eye(V)[y] - p)) for y in range(V))
print("[1b] |E[gg'] - J'F_out J| =", np.abs(Fs - F).max())

# 2. cheapest move against w: d* = -delta F^-1 w / (w'F^-1 w)
Fc = F + 1e-2 * np.trace(F) / d * np.eye(d)
w = rng.normal(size=d)
delta = 0.7
dstar = -delta * np.linalg.solve(Fc, w) / (w @ np.linalg.solve(Fc, w))
best = 0.5 * dstar @ Fc @ dstar
worst_ratio = np.inf
for _ in range(20000):  # random feasible moves never cheaper
    r = rng.normal(size=d)
    r = r - (w @ r + delta) / (w @ w) * w
    worst_ratio = min(worst_ratio, 0.5 * r @ Fc @ r / best)
print(f"[2] constraint w'd*={w@dstar:.4f} (target {-delta}); min random cost / optimum = {worst_ratio:.4f} (>=1)")
print(f"[2b] optimum cost {best:.5f} vs delta^2/(2 w'F^-1 w) {delta**2/(2*w@np.linalg.solve(Fc,w)):.5f}")

# 3. gap transfer formula under dh(x) = -(a + b s_w(x)) M w
n = 4000
M = np.linalg.inv(Fc)
u = rng.normal(size=d)
Hw = rng.normal(size=(n, d)); yw = rng.integers(0, 2, n); Hw[yw == 1] += rng.normal(size=d)
Hu = rng.normal(size=(n, d)); yu = rng.integers(0, 2, n); Hu[yu == 1] += rng.normal(size=d)
a, b = 0.3, 0.05


def edit(H):
    s = H @ w
    return H - np.outer(a + b * s, M @ w)


gap = lambda s, y: s[y == 1].mean() - s[y == 0].mean()
dg_w = gap(edit(Hw) @ w, yw) - gap(Hw @ w, yw)
dg_u = gap(edit(Hu) @ u, yu) - gap(Hu @ u, yu)
Gu, Gw = gap(Hu @ w, yu), gap(Hw @ w, yw)
pred = dg_w * (u @ M @ w) / (w @ M @ w) * Gu / Gw
kappa = -dg_w / Gw
pred_felix = -kappa * (u @ M @ w) / (w @ M @ w) * Gu
print(f"[3] measured dg_u={dg_u:.6f}  ratio-form={pred:.6f}  Felix-form={pred_felix:.6f}")

# 3b. target AUROC unchanged by the affine edit (caveat)
def auroc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = (y == 1).sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print(f"[3b] target AUROC clean={auroc(Hw@w,yw):.4f} edited={auroc(edit(Hw)@w,yw):.4f} (equal: affine, rank-preserving)")
