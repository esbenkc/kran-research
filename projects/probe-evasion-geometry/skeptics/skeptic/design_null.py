"""Design-level null, from PREREG.md text only (no build code).

Exact lemma world (population, no MLP): Sigma spectrum 1/i, d=64, optimal d'=4,
w = mass-mean probe (Delta), 50 probes = 25 logistic(~LDA) + 25 mass-mean in random subspaces of dim
log-uniform [2,64]. Full evasion k=1 by the Mahalanobis minimal move.
Question: how much r^2 does the shuffled-cos null get just from the spread of d'_u?
And what r^2 does raw cosine get when plugged into the same formula?
"""
import numpy as np
from scipy.stats import norm

def run(seed, k=1.0, nperm=1000):
    rng = np.random.default_rng(seed)
    d = 64
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    lam = 1.0 / np.arange(1, d + 1)
    S = R @ np.diag(lam) @ R.T
    D = rng.normal(size=d)
    D *= 4.0 / np.sqrt(D @ np.linalg.solve(S, D))  # optimal d' = 4
    dp = lambda v, Dm: (v @ Dm) / np.sqrt(v @ S @ v)
    w = D.copy()
    U = []
    for j in range(50):
        m = int(np.exp(rng.uniform(np.log(2), np.log(d))))
        B, _ = np.linalg.qr(rng.normal(size=(d, m)))  # random subspace basis
        Ss, Ds = B.T @ S @ B, B.T @ D
        coef = np.linalg.solve(Ss, Ds) if j < 25 else Ds  # LDA-in-subspace vs mass-mean
        U.append(B @ coef)
    U = np.array(U)
    t = k * (w @ D) / (w @ S @ w)
    D2 = D - t * S @ w
    du = np.array([dp(u, D) for u in U]); du2 = np.array([dp(u, D2) for u in U])
    dw = dp(w, D)
    meas = norm.cdf(du / np.sqrt(2)) - norm.cdf(du2 / np.sqrt(2))
    cS = (U @ S @ w) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U) * (w @ S @ w))
    cR = (U @ w) / np.linalg.norm(U, axis=1) / np.linalg.norm(w)
    pred = lambda c: norm.cdf(du / np.sqrt(2)) - norm.cdf((du - k * dw * c) / np.sqrt(2))
    r2 = lambda a, b: np.corrcoef(a, b)[0, 1] ** 2
    null = np.array([r2(meas, pred(rng.permutation(cS))) for _ in range(nperm)])
    return dict(r2_whit=r2(meas, pred(cS)), r2_raw=r2(meas, pred(cR)),
                r2_cos_alone=r2(meas, cS), corr_raw_whit=np.corrcoef(cR, cS)[0, 1],
                null_mean=null.mean(), null_95=np.quantile(null, 0.95),
                refit_auc=norm.cdf(np.sqrt(D2 @ np.linalg.solve(S, D2)) / np.sqrt(2)),
                du_range=(du.min(), du.max()))

for s in range(3):
    print(s, {k: (np.round(v, 3) if not isinstance(v, tuple) else tuple(np.round(v, 2))) for k, v in run(s).items()})
