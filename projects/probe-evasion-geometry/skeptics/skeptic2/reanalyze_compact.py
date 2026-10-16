"""skeptic2: re-analysis from the COMPACT public receipts (data/receipts/*.npz on GitHub).

The full activation arrays are not on GitHub (400 MB). The compact files hold, per run: w, probes,
Sigma_hat, y_test, the 50 probe scores and the w score on clean and encoded TEST data, and the refit
probe's test scores. This script recomputes from those alone: every AUROC, k, the whitened / plain
predictions, both r^2 forms, the nulls (shuffled cos, constant cos, random w), the exact Euclidean
predictor, and the refit AUROC. Sigma is build's Sigma_hat (train, Ledoit-Wolf); `reanalyze.py`
re-estimates it from X_train and gets the same r^2 to 3 decimals (see reanalyze_*.jsonl).

  uv run --project kran-research python <this folder>/reanalyze_compact.py kran-research/projects/probe-evasion-geometry/data/receipts/*.npz
"""
import json
import sys

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(12345)
r2p = lambda a, b: float(np.corrcoef(a, b)[0, 1] ** 2)
r2id = lambda m, p: float(1 - ((m - p) ** 2).sum() / ((m - m.mean()) ** 2).sum())


def analyze(path):
    z = np.load(path)
    y = z["y_test"].astype(int)
    w, U, S = z["w"], z["probes"], z["Sigma_hat"]
    B, A = z["probe_scores_before"].astype(float), z["probe_scores_after"].astype(float)
    eps = 1 / (2 * len(y))
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, eps, 1 - eps))
    A0 = np.array([roc_auc_score(y, B[:, j]) for j in range(B.shape[1])])
    A1 = np.array([roc_auc_score(y, A[:, j]) for j in range(A.shape[1])])
    aw0, aw1 = roc_auc_score(y, z["w_scores_before"]), roc_auc_score(y, z["w_scores_after"])
    dw0 = dp(aw0); k = 1 - dp(aw1) / dw0
    du = dp(A0)
    qu = np.sqrt(np.einsum("ij,jk,ik->i", U, S, U)); qw = np.sqrt(w @ S @ w)
    cw = (U @ S @ w) / (qu * qw)
    cr = (U @ w) / np.linalg.norm(U, axis=1) / np.linalg.norm(w)
    ce = cr * (np.linalg.norm(U, axis=1) / qu) * (qw / np.linalg.norm(w))
    meas = A0 - A1
    pred = lambda c: A0 - norm.cdf((du - k * dw0 * c) / np.sqrt(2))
    pw, pr, pe = pred(cw), pred(cr), pred(ce)
    null1 = np.array([r2p(meas, pred(rng.permutation(cw))) for _ in range(1000)])
    pc = pred(np.full_like(cw, cw.mean()))
    null3 = []
    for _ in range(200):
        wr = rng.normal(size=len(w))
        null3.append(r2p(meas, pred((U @ S @ wr) / (qu * np.sqrt(wr @ S @ wr)))))
    return dict(
        run=str(path).split("/")[-1].removesuffix(".npz"),
        auroc_w=[round(aw0, 4), round(aw1, 4)], k=round(k, 4),
        r2_white=round(r2p(meas, pw), 4), R2id_white=round(r2id(meas, pw), 4),
        r2_raw=round(r2p(meas, pr), 4), r2_euclid_exact=round(r2p(meas, pe), 4),
        r2_cos_white_alone=round(r2p(meas, cw), 4),
        null_shuffled_p95=round(float(np.quantile(null1, 0.95)), 3),
        null_const_cos=round(r2p(meas, pc) if pc.std() > 0 else 0.0, 4),
        null_random_w_p95=round(float(np.quantile(null3, 0.95)), 3),
        refit_auroc=round(roc_auc_score(y, z["refit_scores"]), 4),
        mean_meas_minus_pred=round(float((meas - pw).mean()), 4),
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if "probe_scores_before" not in np.load(p).files:
            print(json.dumps({"run": str(p).split("/")[-1].removesuffix(".npz"), "skipped": "no probe scores in compact receipt"}), flush=True)
            continue
        print(json.dumps(analyze(p)), flush=True)
