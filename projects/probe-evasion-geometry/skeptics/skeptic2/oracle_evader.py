"""skeptic2: what does the BEST possible evader under the frozen loss do? (theory expectation, no training)

Frozen loss (PREREG): lam * d'_w(f(x))^2 + E || Sigma^-1/2 (f(x) - x) ||^2.
The encoder sees x, not y.  Pointwise Lagrangian => the optimal edit is
    f(x) = x + gamma * Sigma w * (pi(x) - p),   pi(x) = P(y=1 | x)  (posterior),
i.e. direction Sigma w (Mahalanobis retain) or w (Euclidean retain), magnitude = centred posterior.
gamma is set so the class-mean gap of w-scores goes to 0 (full evasion, the lam -> inf limit).
Mean gaps then follow the lemma EXACTLY; the within-class covariance changes by Var(pi | y),
which the lemma ignores.  This script measures how much that matters for the prereg r^2.

Design copied from PREREG: d=64, lambda_i ∝ 1/i, random rotation, optimal d'=4,
w = mass-mean probe, 50 probes = 25 logistic + 25 mass-mean on random subspaces dim log-uniform [2,d].
Also runs d'=2 (overlapping classes) to show where the lemma's Sigma-fixed assumption breaks.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def auc(s, y):
    return roc_auc_score(y, s)

def run(seed, dstar=4.0, metric="mahal", d=64, n=20000):
    rng = np.random.default_rng(seed)
    lam = 1.0 / np.arange(1, d + 1)
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    S = R @ np.diag(lam) @ R.T
    v = rng.normal(size=d)
    v *= dstar / np.sqrt(v @ np.linalg.solve(S, v))
    Lc = np.linalg.cholesky(S)
    def sample(m):
        y = np.r_[np.zeros(m // 2), np.ones(m // 2)]
        X = rng.normal(size=(m, d)) @ Lc.T + y[:, None] * v
        return X, y
    Xtr, ytr = sample(n); Xte, yte = sample(n)
    w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    # probe family
    U = []
    for j in range(50):
        k = int(np.round(np.exp(rng.uniform(np.log(2), np.log(d)))))
        P, _ = np.linalg.qr(rng.normal(size=(d, k)))
        Z = Xtr @ P
        if j < 25:
            c = LogisticRegression(C=1.0, max_iter=2000).fit(Z, ytr).coef_[0]
        else:
            c = Z[ytr == 1].mean(0) - Z[ytr == 0].mean(0)
        U.append(P @ c)
    U = np.array(U)
    # oracle posterior (true model) -> best evader
    Si = np.linalg.inv(S)
    def post(X):
        return expit(X @ Si @ v - 0.5 * v @ Si @ v)  # mu0=0, mu1=v, p=1/2
    dirn = S @ w if metric == "mahal" else w
    def enc(X, gamma):
        return X + gamma * np.outer(post(X) - 0.5, dirn)
    # gamma for zero mean gap of w-scores on train
    pi = post(Xtr); G = pi[ytr == 1].mean() - pi[ytr == 0].mean()
    gamma = -(w @ v) / ((w @ dirn) * G)
    Htr, Hte = enc(Xtr, gamma), enc(Xte, gamma)
    Sh = 0.5 * (np.cov(Xtr[ytr == 0].T) + np.cov(Xtr[ytr == 1].T))
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, 1e-6, 1 - 1e-6))
    aw0, aw1 = auc(Xte @ w, yte), auc(Hte @ w, yte)
    k_ = 1 - dp(aw1) / dp(aw0)
    out = []
    for u in U:
        a0, a1 = auc(Xte @ u, yte), auc(Hte @ u, yte)
        cw = u @ Sh @ w / np.sqrt((u @ Sh @ u) * (w @ Sh @ w))
        cr = u @ w / np.linalg.norm(u) / np.linalg.norm(w)
        pw = norm.cdf((dp(a0) - k_ * dp(aw0) * cw) / np.sqrt(2))
        pr = norm.cdf((dp(a0) - k_ * dp(aw0) * cr) / np.sqrt(2))
        out.append((a0 - a1, a0 - pw, a0 - pr, cw, cr))
    o = np.array(out)
    r2 = lambda a, b: np.corrcoef(a, b)[0, 1] ** 2
    refit = LogisticRegression(max_iter=3000).fit(Htr, ytr)
    return dict(aw0=aw0, aw1=aw1, k=k_, r2w=r2(o[:, 0], o[:, 1]), r2r=r2(o[:, 0], o[:, 2]),
                R2w=1 - ((o[:, 0] - o[:, 1]) ** 2).sum() / ((o[:, 0] - o[:, 0].mean()) ** 2).sum(),
                corr_cos=np.corrcoef(o[:, 3], o[:, 4])[0, 1], refit=auc(refit.decision_function(Hte), yte),
                rho_w=(w @ v / np.sqrt(w @ S @ w)) / dstar, drop_range=(o[:, 0].min(), o[:, 0].max()))

if __name__ == "__main__":
    for dstar in [4.0, 2.0]:
        for metric in ["mahal", "euclid"]:
            for seed in range(3):
                r = run(seed, dstar, metric)
                print(f"d*={dstar} {metric:6s} seed{seed}: AUROC_w {r['aw0']:.3f}->{r['aw1']:.3f} k={r['k']:.3f} "
                      f"r2_whit={r['r2w']:.3f} R2_whit(nofit)={r['R2w']:.3f} r2_raw={r['r2r']:.3f} "
                      f"corr(raw,whit cos)={r['corr_cos']:.2f} refit={r['refit']:.3f} rho_w={r['rho_w']:.2f} "
                      f"drop[{r['drop_range'][0]:.3f},{r['drop_range'][1]:.3f}]", flush=True)
