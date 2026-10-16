"""Skeptic's independent re-check from build's raw npz files (uses only X_te, Xe_te, y_te, w, probes).

Recomputes from scratch: AUROC per probe before/after (own rank AUROC), within-class Sigma
(own estimate from clean test, Ledoit-Wolf), d'_u from AUROC (PREREG rule, clipped) and from moments,
k from w, predicted drop with whitened vs raw cosine, squared Pearson r2 (PREREG primary) and
1-SSE/SST (secondary), refit AUROC (own logistic, 2-fold on encoded test), nulls:
  N1 shuffled cos_Sigma across probes (keep d'_u), 2000 perms
  N2 random direction r instead of w in the cosine (keep k), 500 draws
  N3 Sigma = identity (that is raw cos) -- reported as r2_raw
Diagnostics: cosine of class-1-minus-class-0 mean-gap change with Sigma w and with w; ||dSigma||/||Sigma||.
Usage: python check.py file1.npz [file2.npz ...]
"""
import sys, json
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def auroc(s, y):
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_cov(X, y):
    Xc = np.concatenate([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    return LedoitWolf(assume_centered=True).fit(Xc).covariance_


def r2p(a, b):
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def r2id(meas, pred):
    return float(1 - ((meas - pred) ** 2).sum() / ((meas - meas.mean()) ** 2).sum())


def check(path, rng):
    z = np.load(path, allow_pickle=True)
    if "X_test" in z.files:  # decision-run format
        X, Xe, y = z["X_test"], z["E_test"], z["y_test"].astype(int)
        Xtr, Etr, ytr = z["X_train"], z["E_train"], z["y_train"].astype(int)
    else:  # pilot format (test only)
        X, Xe, y = z["X_te"], z["Xe_te"], z["y_te"].astype(int)
        Xtr = Etr = ytr = None
    w, U = z["w"], z["probes"]
    n = len(y)
    eps = 1 / (2 * n)
    dprime = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, eps, 1 - eps))
    # PREREG: Sigma_hat = Ledoit-Wolf pooled within-class on TRAIN; test-set Sigma as robustness
    S_te = within_cov(X, y)
    S = within_cov(Xtr, ytr) if Xtr is not None else S_te
    a_u0 = np.array([auroc(X @ u, y) for u in U]); a_u1 = np.array([auroc(Xe @ u, y) for u in U])
    a_w0, a_w1 = auroc(X @ w, y), auroc(Xe @ w, y)
    du, dw0, dw1 = dprime(a_u0), dprime(a_w0), dprime(a_w1)
    k = 1 - dw1 / dw0
    meas = a_u0 - a_u1

    def cosM(V, r, M):
        return (V @ M @ r) / np.sqrt(np.einsum("ij,jk,ik->i", V, M, V) * (r @ M @ r))

    cS, cR = cosM(U, w, S), cosM(U, w, np.eye(len(w)))
    pred = lambda c: a_u0 - norm.cdf((du - k * dw0 * c) / np.sqrt(2))
    pW, pR = pred(cS), pred(cR)
    null1 = np.array([r2p(meas, pred(rng.permutation(cS))) for _ in range(2000)])
    null2 = []
    for _ in range(500):
        r = rng.normal(size=len(w))
        null2.append(r2p(meas, pred(cosM(U, r, S))))
    null2 = np.array(null2)
    # moment-based d' (alternative to the PREREG AUROC-inversion; checks saturation sensitivity)
    D0 = X[y == 1].mean(0) - X[y == 0].mean(0)
    D1 = Xe[y == 1].mean(0) - Xe[y == 0].mean(0)
    du_m = (U @ D0) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U))
    dw_m = (w @ D0) / np.sqrt(w @ S @ w)
    k_m = 1 - (w @ D1) / (w @ D0)
    pW_m = a_u0 - norm.cdf((du_m - k_m * dw_m * cS) / np.sqrt(2))
    # refit: own logistic. PREREG: fit on encoded train, AUROC on encoded test.
    if Etr is not None:
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(Etr, ytr)
        aucs = [auroc(clf.decision_function(Xe), y)]
        w_massmean_cos = float(np.dot(w, Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))
                               / np.linalg.norm(w) / np.linalg.norm(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)))
    else:
        aucs = []
        for tr, te in StratifiedKFold(2, shuffle=True, random_state=0).split(Xe, y):
            clf = LogisticRegression(max_iter=2000).fit(Xe[tr], y[tr])
            aucs.append(auroc(clf.decision_function(Xe[te]), y[te]))
        w_massmean_cos = None
    pW_te = pred(cosM(U, w, S_te))
    # diagnostics
    dD = D1 - D0
    Sw = S @ w
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    Se = within_cov(Xe, y)
    return dict(
        file=path.split("/")[-1], n_test=int(n), k=float(k), auc_w=(float(a_w0), float(a_w1)),
        n_probes_auc_gt_0995=int((a_u0 > 0.995).sum()),
        r2_whit=r2p(meas, pW), r2_raw=r2p(meas, pR),
        R2id_whit=r2id(meas, pW), R2id_raw=r2id(meas, pR),
        r2_whit_moment_dprime=r2p(meas, pW_m),
        r2_cos_alone=r2p(meas, cS), corr_raw_whit=float(np.corrcoef(cR, cS)[0, 1]),
        null_shuffle_mean=float(null1.mean()), null_shuffle_p95=float(np.quantile(null1, .95)),
        null_shuffle_p=float((null1 >= r2p(meas, pW)).mean()),
        null_randdir_mean=float(null2.mean()), null_randdir_p95=float(np.quantile(null2, .95)),
        refit_auc=float(np.mean(aucs)),
        cos_shift_Sw=cos(dD, Sw), cos_shift_w=cos(dD, w),
        rel_dSigma=float(np.linalg.norm(Se - S_te) / np.linalg.norm(S_te)),
        r2_whit_testSigma=r2p(meas, pW_te), w_is_massmean_cos=w_massmean_cos,
        sigma_match_build=(float(np.linalg.norm(S - z[sk]) / np.linalg.norm(z[sk]))
                           if (sk := ("Sigma_hat" if "Sigma_hat" in z.files else "Sigma")) in z.files else None),
        meas_drop_range=(float(meas.min()), float(meas.max())),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    out = [check(p, rng) for p in sys.argv[1:]]
    for o in out:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}))
    for key in ("r2_whit", "r2_raw", "refit_auc", "R2id_whit", "R2id_raw", "null_shuffle_p95"):
        print(f"MEAN {key}: {np.mean([o[key] for o in out]):.4f}")
