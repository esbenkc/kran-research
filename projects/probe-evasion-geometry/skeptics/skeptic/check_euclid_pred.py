"""Own check of the post's footnote [^euclid]: exact predictor for a Euclidean move Delta -> Delta - t w.
d'_u -> d'_u - k d'_w cos(u,w) |u| sqrt(w'Sw) / (|w| sqrt(u'Su)).
Derivation: u.(t w) / sqrt(u'Su) with t = k (w.Delta)/(w.w); d'_w = w.Delta/sqrt(w'Sw)
=> t u.w / sqrt(u'Su) = k d'_w sqrt(w'Sw) (u.w) / (w.w sqrt(u'Su)) = k d'_w cos(u,w) |u| sqrt(w'Sw)/(|w| sqrt(u'Su)).
"""
import sys
import numpy as np
from scipy.stats import norm
from check import auroc, within_cov, r2p

for p in sys.argv[1:]:
    z = np.load(p, allow_pickle=True)
    X, Xe, y = z["X_test"], z["E_test"], z["y_test"].astype(int)
    S = within_cov(z["X_train"], z["y_train"].astype(int))
    w, U = z["w"], z["probes"]
    eps = 1 / (2 * len(y))
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, eps, 1 - eps))
    a0 = np.array([auroc(X @ u, y) for u in U]); a1 = np.array([auroc(Xe @ u, y) for u in U])
    dw0, dw1 = dp(auroc(X @ w, y)), dp(auroc(Xe @ w, y)); k = 1 - dw1 / dw0
    cR = U @ w / np.linalg.norm(U, axis=1) / np.linalg.norm(w)
    fac = np.linalg.norm(U, axis=1) * np.sqrt(w @ S @ w) / (np.linalg.norm(w) * np.sqrt(np.einsum("ij,jk,ik->i", U, S, U)))
    pred = a0 - norm.cdf((dp(a0) - k * dw0 * cR * fac) / np.sqrt(2))
    print(p.split("/")[-1], "exact-Euclid predictor r2", round(r2p(a0 - a1, pred), 4))
