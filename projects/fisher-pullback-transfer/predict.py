"""The frozen transfer predictor (PREREG.md, 'The predictor').

For target w and probe u, on u's CLEAN test set:
    s_u_pred(x) = s_u(x) + ratio * r_P(u, w) * s_w(x),   ratio = delta_w / G_w(w),   r_P = u'Pw / w'Pw
predicted drop = AUROC_clean(u) - AUROC(s_u_pred)   (empirical AUROC, primary).
P = I (plain), Sigma_LW (whitened), (F_common + lambda I)^-1 with lambda = c * median eig(F) (Fisher).
F_common is the 8000-window estimate (data/fisher/fisher_big.npz, DEVIATIONS 15:14); the 2000-window one in
fisher_sigma.npz is a sensitivity row.
Shared-factor baseline: r_P := 1.  Gaussian map: sensitivity row only.
Scores use s_u = u'h without the bias (AUROC ignores it).
"""
import json

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from common import DATA

DAMPING_C = 1.0


def load_clean():
    meta = json.loads((DATA / "probes" / "probes.json").read_text())
    Wt = np.load(DATA / "probes" / "probe_weights.npz")
    probes = [dict(p, w=Wt["W"][p["idx"]]) for p in meta]
    acts = {}
    for p in probes:
        s = p["source"]
        if s not in acts:
            z = np.load(DATA / "clean" / f"acts_{s}.npz")
            te = z["split"] == 1
            acts[s] = {"X": z["X"][te].astype(np.float64), "label": z["label"][te]}
    return {"probes": probes, "acts": acts}


def labels(clean, u):
    return (clean["acts"][u["source"]]["label"] == u["label"]).astype(int)


def fisher_metric(c=DAMPING_C, which="F_common", file="fisher_big.npz"):
    z = np.load(DATA / "fisher" / file)
    F = z[which]
    lam = c * np.median(np.linalg.eigvalsh(F))
    return np.linalg.inv(F + lam * np.eye(F.shape[0]))


def metrics(c=DAMPING_C):
    z = np.load(DATA / "fisher" / "fisher_sigma.npz")
    return {"plain": np.eye(z["Sigma_LW"].shape[0]), "whitened": z["Sigma_LW"], "fisher": fisher_metric(c)}


def r_P(u, w, P):
    return 1.0 if P is None else float(u @ P @ w) / float(w @ P @ w)


def predicted_drop(clean, u, w, P, ratio):
    X = clean["acts"][u["source"]]["X"]
    y = labels(clean, u)
    su, sw = X @ u["w"], X @ w["w"]
    return roc_auc_score(y, su) - roc_auc_score(y, su + ratio * r_P(u["w"], w["w"], P) * sw)


def predicted_drop_gauss(clean, u, w, P, ratio):
    X = clean["acts"][u["source"]]["X"]
    y = labels(clean, u)
    su, sw = X @ u["w"], X @ w["w"]
    gap = su[y == 1].mean() - su[y == 0].mean()
    Gu = sw[y == 1].mean() - sw[y == 0].mean()
    sd = np.sqrt(0.5 * (su[y == 1].var(ddof=1) + su[y == 0].var(ddof=1)))
    shift = ratio * r_P(u["w"], w["w"], P) * Gu
    return norm.cdf(gap / sd / np.sqrt(2)) - norm.cdf((gap + shift) / sd / np.sqrt(2))
