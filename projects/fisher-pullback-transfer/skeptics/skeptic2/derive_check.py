"""Hugo's own numeric checks of the three derivation steps (no build code used).
S1: KL(p(h) || p(h+d)) = 1/2 d' J' (diag p - p p') J d + O(|d|^3) for logits z = g(h).
S2: argmin 1/2 d'(F+lam I)d s.t. w'd = -eps is d* = -eps P w / (w'P w), P = (F+lam I)^-1.
S3: edit dh = -(a + b s_w) P w with b fixed by the target gap change delta_w gives
    s_u' = s_u + (delta_w/G_w) r_P s_w + const, r_P = u'Pw / w'Pw; AUROC of the target is unchanged
    when 1 + delta_w/G_w > 0 and flips to 1 - AUROC when it is < 0.
"""
import numpy as np
rng = np.random.default_rng(0)
d, V = 12, 30
A1 = rng.standard_normal((d, d)) / np.sqrt(d); A2 = rng.standard_normal((V, d)) / np.sqrt(d)
def z(h): return A2 @ np.tanh(A1 @ h)
def sm(x): e = np.exp(x - x.max()); return e / e.sum()
h = rng.standard_normal(d)
J = A2 @ np.diag(1 - np.tanh(A1 @ h) ** 2) @ A1
p = sm(z(h)); Fo = np.diag(p) - np.outer(p, p); F = J.T @ Fo @ J
for eps in (1e-1, 1e-2, 1e-3):
    dd = eps * rng.standard_normal(d); q = sm(z(h + dd))
    kl = (p * (np.log(p) - np.log(q))).sum()
    print(f"S1 eps={eps:g} KL/(0.5 d'Fd) = {kl / (0.5 * dd @ F @ dd):.6f}")
# S2
lam = np.median(np.linalg.eigvalsh(F)); M = F + lam * np.eye(d); P = np.linalg.inv(M)
w = rng.standard_normal(d); eps = 0.3
ds = -eps * P @ w / (w @ P @ w)
best = 0.5 * ds @ M @ ds
worst_gap = 0
for _ in range(20000):
    r = rng.standard_normal(d); r -= (w @ r) / (w @ w) * w   # feasible perturbation keeps w'd fixed
    dd = ds + 0.1 * r
    worst_gap = min(worst_gap, 0.5 * dd @ M @ dd - best)
print(f"S2 constraint w'd* = {w @ ds:.6f} (want {-eps}); min over 20000 feasible perturbations of cost - cost* = {worst_gap:.3e} (>= 0)")
# S3
n = 400; X = rng.standard_normal((n, d)); y = (rng.random(n) < 0.5).astype(int); X[y == 1] += 0.4 * w / np.linalg.norm(w)
u = rng.standard_normal(d); a = 0.7
sw = X @ w; su = X @ u
b = 0.5 / ((w @ P @ w) * (sw[y == 1].mean() - sw[y == 0].mean()))  # delta_w = -0.5 (no flip)
Xe = X - np.outer(a + b * sw, P @ w)
Gw = sw[y == 1].mean() - sw[y == 0].mean(); swe = Xe @ w
dw = (swe[y == 1].mean() - swe[y == 0].mean()) - Gw
pred = su + dw / Gw * (u @ P @ w) / (w @ P @ w) * sw
diff = Xe @ u - pred
print(f"S3 delta_w formula -b w'Pw G_w = {-b * (w @ P @ w) * Gw:.6f} vs measured {dw:.6f}; edited s_u - predicted is constant: sd = {diff.std():.2e}")
def auroc(s, y):
    s1, s0 = s[y == 1], s[y == 0]
    return ((s1[:, None] > s0[None]).mean() + 0.5 * (s1[:, None] == s0[None]).mean())
print(f"S3 target AUROC clean {auroc(sw, y):.4f}, edited {auroc(swe, y):.4f}, 1+ratio = {1 + dw / Gw:.3f}")
b2 = 1.5 / (w @ P @ w); swe2 = sw - (a + b2 * sw) * (w @ P @ w)
print(f"S3 with 1+ratio < 0: edited target AUROC {auroc(swe2, y):.4f} = 1 - clean {1 - auroc(sw, y):.4f}")
