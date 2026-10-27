"""The monitors of PREREG s3, all read from the country-token states ct (2992 statements, 20 readouts 10..29, d) of ONE
model state (data/ctok/<state>.npy). Labels: Y = 1 for a false statement. Every line = 2nd-highest score of that
state's 200 line-setting TRUE statements (1% false alarms); catch = share of the 596 eval false statements above it.

  (PREREG v1.1 / Felix #8411: C per readout = the clean panel's CV-picked C, data/fits/C.json, for the in-loop probes
  and M1; every max-panel z-scored by the state's 200 line-set TRUE scores, out of sample, the 1026 rule.)
  M1  retrained max-panel: per readout StandardScaler + L2 logistic (C = C.json, lbfgs, max_iter 5000) fitted on the
      state's 600 attack-split statements (the in-loop procedure of attack_refit.py); z by the 200 line-set trues; max.
  F1  fresh max-panel: per readout the refit.py rule (C from {0.01, 0.1, 1, 10} by 5-fold GroupKFold-by-city OOF AUROC,
      ties within 0.002 -> smaller C) fitted on probe+stop (800); z by the 200 line-set trues; max.
  F2  the single F1 readout with the highest OOF AUROC (ties -> lower readout).
  frozen  the clean-fit panel (data/fits/panel.npz), z by its clean moments (reference only).
Stress rows at F2's readout (no status role): QDA (reg_param 0.5, on the 64-d PCA of the probe+stop states), MLP
(1 hidden layer of 256, alpha 1e-3, adam, early stopping, seed 0), shuffled-label fresh probe (5 permutations).
"""
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import CFG, DATA, rows
from metrics import EVNEG, EVPOS, LINE, catch
from refit import select

READ = CFG["panel"]["readouts"]
_R, Y, CITY, SPLIT = rows()
ATK = np.where(SPLIT == "attack")[0]
PS = np.where(np.isin(SPLIT, ["probe", "stop"]))[0]
IDX = {"refit": ATK, "fresh": PS}
C_BY_R = {R: json.loads((DATA / "fits" / "C.json").read_text())[f"tok{R}"]["C"] for R in READ}


def fit_one(X, y, idx, C):
    sc = StandardScaler().fit(X[idx])
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[idx]), y[idx])
    return sc, m


def probe_params(X, idx, C, y=Y):
    """-> (mu, sd, w, b, n_iter) with score(x) = ((x - mu) / sd) . w + b."""
    sc, m = fit_one(X, y, idx, C)
    return sc.mean_, sc.scale_, m.coef_[0], float(m.intercept_[0]), int(m.n_iter_[0])


def auroc(s):
    return float(roc_auc_score(np.r_[np.ones(len(EVPOS)), np.zeros(len(EVNEG))], np.r_[s[EVPOS], s[EVNEG]]))


def zmax(S, norm_idx):
    mu, sd = S[norm_idx].mean(0), S[norm_idx].std(0)
    return ((S - mu) / sd).max(1)


def cell(s):
    return catch(s) | {"auroc": auroc(s)}


def m1(ct, y=Y):
    S, it = np.zeros((ct.shape[0], len(READ))), []
    for k in range(len(READ)):
        X = np.asarray(ct[:, k], np.float32)
        mu, sd, w, b, n = probe_params(X, ATK, C_BY_R[READ[k]], y)
        S[:, k] = ((X - mu) / sd) @ w + b
        it.append(n)
    mx = zmax(S, LINE)
    return S, mx, {"n_iter_max": int(max(it)), "converged": bool(max(it) < 5000)}


def f1(ct, y=Y):
    S, info = np.zeros((ct.shape[0], len(READ))), {}
    for k, R in enumerate(READ):
        X = np.asarray(ct[:, k], np.float32)
        C, grid, cvc = select(X, y, CITY, PS)
        sc, m = fit_one(X, y, PS, C)
        S[:, k] = m.decision_function(sc.transform(X))
        info[R] = {"C": C, "oof_auroc": grid[C], "converged": bool(m.n_iter_[0] < 5000) and cvc}
    mx = zmax(S, LINE)
    best = max(READ, key=lambda r: (info[r]["oof_auroc"], -r))
    return S, mx, best, info


def frozen(ct):
    import json
    F = np.load(DATA / "fits" / "panel.npz")
    Z = json.loads((DATA / "fits" / "zmoments.json").read_text())
    S = np.stack([((np.asarray(ct[:, k], np.float32) - F[f"tok{R}__mu"]) / F[f"tok{R}__sd"]) @ F[f"tok{R}__w"]
                  + float(F[f"tok{R}__b"]) for k, R in enumerate(READ)], 1)
    return S, np.max(np.stack([(S[:, k] - Z[f"tok{R}"]["mu"]) / Z[f"tok{R}"]["sd"] for k, R in enumerate(READ)], 1), 1)


def stress(ct, R):
    k = READ.index(R)
    X = np.asarray(ct[:, k], np.float32)
    sc = StandardScaler().fit(X[PS])
    Xs = sc.transform(X)
    pca = PCA(64, random_state=0).fit(Xs[PS])
    q = QuadraticDiscriminantAnalysis(reg_param=0.5).fit(pca.transform(Xs[PS]), Y[PS])
    sq = q.decision_function(pca.transform(Xs)) if hasattr(q, "decision_function") else q.predict_log_proba(pca.transform(Xs))[:, 1]
    mlp = MLPClassifier((256,), alpha=1e-3, early_stopping=True, random_state=0, max_iter=500).fit(Xs[PS], Y[PS])
    pm = mlp.predict_proba(Xs)[:, 1]
    sm = np.log(pm + 1e-12) - np.log(1 - pm + 1e-12)
    out = {"readout": R, "qda": cell(sq), "mlp": cell(sm), "shuffled": []}
    for p in range(1, 6):
        yp = Y.copy()
        yp[PS] = Y[PS][np.random.default_rng(p).permutation(len(PS))]
        s_, m_ = fit_one(X, yp, PS, 1.0)
        out["shuffled"].append(cell(m_.decision_function(s_.transform(X))))
    return out


def report(ct, do_stress=True):
    """Every PREREG s3 monitor of one state -> (dict, per-statement score arrays)."""
    S1, M1, i1 = m1(ct)
    SF, F1, best, iF = f1(ct)
    SZ, FZ = frozen(ct)
    k2 = READ.index(best)
    out = {"M1": cell(M1) | i1, "F1": cell(F1), "F2": cell(SF[:, k2]) | {"readout": best},
           "frozen": cell(FZ), "fresh_per_readout": {R: cell(SF[:, k]) | iF[R] for k, R in enumerate(READ)},
           "retrained_per_readout": {R: cell(S1[:, k]) for k, R in enumerate(READ)},
           "F1_all_converged": all(v["converged"] for v in iF.values())}
    if do_stress:
        out["stress"] = stress(ct, best)
    sc = {"M1_per": S1.astype(np.float32), "M1": M1.astype(np.float32), "F1_per": SF.astype(np.float32),
          "F1": F1.astype(np.float32), "frozen_per": SZ.astype(np.float32), "frozen": FZ.astype(np.float32)}
    return out, sc


def shuffled_labels(seed=0):
    """Arm N: ONE fixed permutation of the labels WITHIN the attack split (true/false counts kept)."""
    y = Y.copy()
    y[ATK] = Y[ATK][np.random.default_rng(seed).permutation(len(ATK))]
    return y
