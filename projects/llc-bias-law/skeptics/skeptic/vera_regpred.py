"""Price Tove/Iris's zero-parameter regular-direction localisation bias -1/2 sum_i gamma/(nbeta*h_i+gamma)
(h_i = nonzero Hessian eigenvalues of K at w*) against the measured bias in vera_isoG/isoX (population RRR)."""
import json, numpy as np
M = N = 32; H = 16
def hess_eigs(r, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, r))); V, _ = np.linalg.qr(rng.standard_normal((M, r)))
    s = np.linspace(1.0, 2.0, r)
    B0 = np.zeros((N, H)); A0 = np.zeros((H, M)); B0[:, :r] = U*np.sqrt(s); A0[:r] = (V*np.sqrt(s)).T
    # Jacobian of vec(BA) wrt (B, A): Gauss-Newton Hessian = J^T J (residual is 0 at w*)
    J = np.zeros((N*M, N*H + H*M))
    for i in range(N):
        for k in range(H):
            E = np.zeros((N, H)); E[i, k] = 1; J[:, i*H+k] = (E @ A0).ravel()
    for k in range(H):
        for j in range(M):
            E = np.zeros((H, M)); E[k, j] = 1; J[:, N*H + k*M + j] = (B0 @ E).ravel()
    ev = np.linalg.eigvalsh(J.T @ J)
    return ev[ev > 1e-9]
res = {}
for r, seed in ((1, 1), (8, 1)):
    h = hess_eigs(r, seed)
    for g in (0.01, 0.1, 1.0):
        pred = -0.5*np.sum(g/(h+g))   # gamma/nbeta = g, so gamma/(nbeta h+gamma) = g/(h+g)
        res[f"r{r}_g{g}"] = dict(n_regular=len(h), h_min=float(h.min()), h_max=float(h.max()), pred_abs=float(pred))
        print(f"r={r} nonzero eigs={len(h)} h in [{h.min():.2f},{h.max():.2f}] g/nb={g}: regular-dir bias {pred:+.2f}")
json.dump(res, open("vera_regpred.json", "w"), indent=1)
