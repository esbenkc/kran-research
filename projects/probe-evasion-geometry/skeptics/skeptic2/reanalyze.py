"""skeptic2: independent re-analysis of build's raw data (own code; only raw arrays are read).

Usage:  uv run --project kran-research python swarm-theory-1016-probe-evasion/skeptic2/reanalyze.py <npz> [<npz> ...]

From each npz it reads raw activations + probes only:
  X_te, Xe_te, y_te, w, probes   (required)
  X_tr, Xe_tr, y_tr              (optional; if absent, Sigma and refit come from a 50/50 split of test)
Nothing from build's per-probe tables is used.

Per file it reports (PREREG rules, own implementation):
  - AUROC before/after for w and every probe (sklearn roc_auc_score)
  - Sigma_hat = Ledoit-Wolf pooled within-class covariance (train if present)
  - d'_u = sqrt2 * Phi^-1(AUROC_u before), clipped to [1/(2n), 1-1/(2n)]
  - k = 1 - d'_w(after)/d'_w(before)
  - predicted drop (whitened) / (plain cosine); squared Pearson r^2 (primary) and 1-SSE/SST
  - refit: fresh logistic probe on encoded train, AUROC on encoded test
Nulls / alternatives:
  N1 shuffled-cos: permute cos_Sigma across probes, keep d'_u (1000 perms) -> null r^2 distribution
  N2 constant-cos: every probe gets the mean cos_Sigma -> r^2 from baseline spread alone
  N3 random-w: compute cos_Sigma against a random direction instead of w (200 draws)
  N4 exact Euclidean-move predictor (what an L2-paying evader would give)
  bootstrap 95% CI of whitened r^2 over probes (2000 resamples)
Diagnostics: class-mean-gap shift cosine with Sigma w and with w; relative covariance change.
"""
import json
import sys

import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(12345)


def pooled_lw(X, y):
    Xc = np.concatenate([X[y == 0] - X[y == 0].mean(0), X[y == 1] - X[y == 1].mean(0)])
    return LedoitWolf(assume_centered=True).fit(Xc).covariance_


def r2p(a, b):
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def r2id(m, p):
    return float(1 - ((m - p) ** 2).sum() / ((m - m.mean()) ** 2).sum())


def analyze(path):
    z = dict(np.load(path, allow_pickle=True))
    alias = {"X_train": "X_tr", "y_train": "y_tr", "E_train": "Xe_tr", "X_test": "X_te", "y_test": "y_te", "E_test": "Xe_te"}
    for a, b in alias.items():
        if a in z and b not in z:
            z[b] = z[a]
    X_te, Xe_te, y_te = z["X_te"].astype(float), z["Xe_te"].astype(float), z["y_te"].astype(int)
    w, U = z["w"].astype(float), z["probes"].astype(float)
    if "X_tr" in z and "Xe_tr" in z:
        X_tr, Xe_tr, y_tr = z["X_tr"].astype(float), z["Xe_tr"].astype(float), z["y_tr"].astype(int)
        X_ev, Xe_ev, y_ev = X_te, Xe_te, y_te
        src = "train/test as saved"
    else:  # split test: first half -> Sigma + refit fit, second half -> evaluation
        idx = rng.permutation(len(y_te)); h = len(idx) // 2
        X_tr, Xe_tr, y_tr = X_te[idx[:h]], Xe_te[idx[:h]], y_te[idx[:h]]
        X_ev, Xe_ev, y_ev = X_te[idx[h:]], Xe_te[idx[h:]], y_te[idx[h:]]
        src = "no train arrays: 50/50 split of test"
    n = len(y_ev)
    eps = 1 / (2 * n)
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, eps, 1 - eps))
    S = pooled_lw(X_tr, y_tr)

    A0 = np.array([roc_auc_score(y_ev, X_ev @ u) for u in U])
    A1 = np.array([roc_auc_score(y_ev, Xe_ev @ u) for u in U])
    aw0, aw1 = roc_auc_score(y_ev, X_ev @ w), roc_auc_score(y_ev, Xe_ev @ w)
    dw0 = dp(aw0); k = 1 - dp(aw1) / dw0
    du = dp(A0)
    cw = (U @ S @ w) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U) * (w @ S @ w))
    cr = (U @ w) / np.linalg.norm(U, axis=1) / np.linalg.norm(w)
    meas = A0 - A1
    pred = lambda c: A0 - norm.cdf((du - k * dw0 * c) / np.sqrt(2))
    pw, pr = pred(cw), pred(cr)

    # N1 shuffled cos
    null1 = np.array([r2p(meas, pred(rng.permutation(cw))) for _ in range(1000)])
    # N2 constant cos
    pc = pred(np.full_like(cw, cw.mean()))
    r2_const = r2p(meas, pc) if pc.std() > 0 else 0.0
    # N3 random w
    null3 = []
    for _ in range(200):
        wr = rng.normal(size=len(w))
        c3 = (U @ S @ wr) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U) * (wr @ S @ wr))
        null3.append(r2p(meas, pred(c3)))
    null3 = np.array(null3)
    # N4 exact Euclidean-move predictor: d'_u drop = k*d'_w * cos(u,w) * (|u|/sqrt(u'Su)) * (sqrt(w'Sw)/|w|)
    qu = np.sqrt(np.einsum("ij,jk,ik->i", U, S, U)); qw = np.sqrt(w @ S @ w)
    ce = cr * (np.linalg.norm(U, axis=1) / qu) * (qw / np.linalg.norm(w))
    pe = pred(ce)
    # bootstrap CI
    bs = []
    for _ in range(2000):
        b = rng.integers(0, len(U), len(U))
        if meas[b].std() > 0 and pw[b].std() > 0:
            bs.append(r2p(meas[b], pw[b]))
    # refit
    clf = LogisticRegression(max_iter=5000).fit(Xe_tr, y_tr)
    refit = roc_auc_score(y_ev, clf.decision_function(Xe_ev))
    clf0 = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    base = roc_auc_score(y_ev, clf0.decision_function(X_ev))
    # diagnostics
    g0 = X_tr[y_tr == 1].mean(0) - X_tr[y_tr == 0].mean(0)
    g1 = Xe_tr[y_tr == 1].mean(0) - Xe_tr[y_tr == 0].mean(0)
    sh = g1 - g0
    cosv = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    S1 = pooled_lw(Xe_tr, y_tr)
    return dict(
        file=str(path), source=src, n_eval=n, n_probes=len(U),
        auroc_w=(round(aw0, 4), round(aw1, 4)), k=round(k, 4),
        r2_white=round(r2p(meas, pw), 4), R2id_white=round(r2id(meas, pw), 4),
        r2_raw=round(r2p(meas, pr), 4), R2id_raw=round(r2id(meas, pr), 4),
        r2_white_CI95=[round(float(np.quantile(bs, q)), 3) for q in (0.025, 0.975)],
        null_shuffled_cos_mean_p95=[round(float(null1.mean()), 3), round(float(np.quantile(null1, 0.95)), 3)],
        null_shuffled_cos_frac_ge_obs=float((null1 >= r2p(meas, pw)).mean()),
        null_const_cos_r2=round(r2_const, 4),
        null_random_w_mean_p95=[round(float(null3.mean()), 3), round(float(np.quantile(null3, 0.95)), 3)],
        r2_euclid_exact=round(r2p(meas, pe), 4), R2id_euclid_exact=round(r2id(meas, pe), 4),
        r2_cos_white_alone=round(r2p(meas, cw), 4),
        corr_raw_white_cos=round(float(np.corrcoef(cr, cw)[0, 1]), 3),
        n_probes_auroc_gt_0995=int((A0 > 0.995).sum()),
        refit_auroc=round(refit, 4), clean_logistic_auroc=round(base, 4),
        shift_cos_Sigma_w=round(cosv(sh, S @ w), 4), shift_cos_w=round(cosv(sh, w), 4),
        cov_change_rel=round(float(np.linalg.norm(S1 - S) / np.linalg.norm(S)), 4),
        drop_range=[round(float(meas.min()), 3), round(float(meas.max()), 3)],
    )


if __name__ == "__main__":
    res = [analyze(p) for p in sys.argv[1:]]
    for r in res:
        print(json.dumps(r), flush=True)
