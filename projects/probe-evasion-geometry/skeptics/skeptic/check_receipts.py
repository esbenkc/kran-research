"""Recompute the headline numbers from the PUBLIC compact receipts (data/receipts/*.npz), so anyone can
re-run the skeptic check without the 400 MB local arrays. Sigma_hat comes from the receipt here (check.py
re-estimates it from X_train and matched it to 0.0 relative error)."""
import sys
import numpy as np
from scipy.stats import norm
from check import auroc, r2p

for p in sys.argv[1:]:
    z = np.load(p, allow_pickle=True)
    y, S, w, U = z["y_test"].astype(int), z["Sigma_hat"], z["w"], z["probes"]
    eps = 1 / (2 * len(y))
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, eps, 1 - eps))
    a0 = np.array([auroc(s, y) for s in z["probe_scores_before"].T])
    a1 = np.array([auroc(s, y) for s in z["probe_scores_after"].T])
    dw0, dw1 = dp(auroc(z["w_scores_before"], y)), dp(auroc(z["w_scores_after"], y))
    k = 1 - dw1 / dw0
    cosM = lambda M: (U @ M @ w) / np.sqrt(np.einsum("ij,jk,ik->i", U, M, U) * (w @ M @ w))
    pred = lambda c: a0 - norm.cdf((dp(a0) - k * dw0 * c) / np.sqrt(2))
    print(p.split("/")[-1], "whit r2 %.4f raw r2 %.4f refit %.4f" % (
        r2p(a0 - a1, pred(cosM(S))), r2p(a0 - a1, pred(cosM(np.eye(len(w))))), auroc(z["refit_scores"], y)))
