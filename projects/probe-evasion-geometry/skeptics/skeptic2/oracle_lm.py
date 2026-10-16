"""skeptic2: theory expectation on the REAL LM activations (pythia-70m L3), before build's LM evasion run.

Applies the analytically optimal evader for the frozen loss (Mahalanobis per-sample retain, full
evasion of the mass-mean probe w):  f(x) = x + gamma * Sigma_hat w * (pi_hat(x) - p),
pi_hat = logistic posterior fit on clean train. Then runs the PREREG analysis on test.
Also the label-aware pure mean shift (the lemma's exact world): x + y * (-t Sigma_hat w).
"""
import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P = "kran-research/projects/probe-evasion-geometry/data/lm_acts_L3.npz"
z = np.load(P, allow_pickle=True)
Xtr, ytr, Xte, yte = (z[k].astype(float) if k[0] == "X" else z[k].astype(int) for k in ["X_train", "y_train", "X_test", "y_test"])
d = Xtr.shape[1]
Xc = np.concatenate([Xtr[ytr == c] - Xtr[ytr == c].mean(0) for c in (0, 1)])
S = LedoitWolf(assume_centered=True).fit(Xc).covariance_
w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
post = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, ytr)
eps = 1 / (2 * len(yte))
dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, eps, 1 - eps))
r2 = lambda a, b: np.corrcoef(a, b)[0, 1] ** 2

for seed in range(3):
    rng = np.random.default_rng(seed)
    U = []
    for j in range(50):
        k = int(np.round(np.exp(rng.uniform(np.log(2), np.log(d)))))
        Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
        Z = Xtr @ Q
        c = LogisticRegression(C=1.0, max_iter=3000).fit(Z, ytr).coef_[0] if j < 25 else Z[ytr == 1].mean(0) - Z[ytr == 0].mean(0)
        U.append(Q @ c)
    U = np.array(U)
    Sw = S @ w
    def evaluate(Htr, Hte, name):
        A0 = np.array([roc_auc_score(yte, Xte @ u) for u in U]); A1 = np.array([roc_auc_score(yte, Hte @ u) for u in U])
        aw0, aw1 = roc_auc_score(yte, Xte @ w), roc_auc_score(yte, Hte @ w)
        kk = 1 - dp(aw1) / dp(aw0)
        cw = U @ S @ w / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U) * (w @ S @ w))
        cr = U @ w / np.linalg.norm(U, axis=1) / np.linalg.norm(w)
        pr = lambda c: A0 - norm.cdf((dp(A0) - kk * dp(aw0) * c) / np.sqrt(2))
        m = A0 - A1
        ref = LogisticRegression(C=1.0, max_iter=5000).fit(Htr, ytr)
        print(f"seed{seed} {name}: AUROC_w {aw0:.3f}->{aw1:.3f} k={kk:.2f} r2_white={r2(m, pr(cw)):.3f} r2_raw={r2(m, pr(cr)):.3f} "
              f"corr(raw,white)={np.corrcoef(cr, cw)[0,1]:.2f} refit={roc_auc_score(yte, ref.decision_function(Hte)):.3f} "
              f"drop[{m.min():.3f},{m.max():.3f}] probes>0.995: {(A0>0.995).sum()}", flush=True)
    # (a) posterior-oracle evader (sees x only)
    pitr, pite = post.predict_proba(Xtr)[:, 1], post.predict_proba(Xte)[:, 1]
    G = pitr[ytr == 1].mean() - pitr[ytr == 0].mean()
    gamma = -(w @ w) / ((w @ Sw) * G)   # mean gap of w-scores on train is w.w
    evaluate(Xtr + gamma * np.outer(pitr - 0.5, Sw), Xte + gamma * np.outer(pite - 0.5, Sw), "posterior-oracle")
    # (b) label-aware mean shift (lemma's exact world)
    t = (w @ w) / (w @ Sw)
    evaluate(Xtr - t * np.outer(ytr, Sw), Xte - t * np.outer(yte, Sw), "label-aware shift")
