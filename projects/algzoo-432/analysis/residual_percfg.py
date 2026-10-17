"""T16 (Iris): oracle candidate-set ceilings + per-config piecewise-linear comparators. FITTED, discovery only.

Config = (pM, pS, pT) in the phantom-augmented order (phantom 0 at position -1, decodes to output 0).
Per config, least squares of the model's logit difference (candidate minus S) on a feature bank, fit only on
near-ties (|v_cand - v_S| < 0.1 * rms(x)). Decision: argmax over candidates of the predicted difference
(S has 0). Train dev seed 81 (2e6), test dev seed 82 (1e6). Never the holdout.
Usage: .venv/bin/python residual_percfg.py   (~35 s). Import fit_percfg() to reuse a fitted predictor.
"""
import json
import numpy as np
from residual_common import data, aug_order


def rec_feats(X):
    n = len(X); R = np.zeros(n); S2 = np.full(n, -1e9); Rs, Ss = [], []
    for t in range(10):
        x = X[:, t]
        S2 = np.where(x > R, R, np.maximum(S2, x)); R = np.maximum(R, x)
        Rs.append(R.copy()); Ss.append(np.maximum(S2, -5))
    return np.stack(Rs, 1), np.stack(Ss, 1)


def bank_full(X):
    R, S2 = rec_feats(X)
    return np.c_[X, R[:, :9], S2[:, 1:9], np.maximum(X, 0)]


BANKS = {"x": lambda X: X, "x,relu(x),running max,running 2nd": bank_full}


def cfg_id(o):
    return (o[:, 0] + 1) * 121 + (o[:, 1] + 1) * 11 + (o[:, 2] + 1)


def fit_percfg(Xa, La, bank, cands):
    """Fit per-config linear predictors of L[cand] - L[S]; returns (predict(X) -> positions, n_coefs)."""
    fn = BANKS[bank]
    oa = aug_order(Xa); Oa = np.maximum(oa, 0); ca = cfg_id(oa); ara = np.arange(len(Xa))
    Fa = fn(Xa)
    va = np.take_along_axis(np.c_[np.zeros(len(Xa)), Xa], oa + 1, 1)
    sca = np.sqrt((Xa ** 2).mean(1))
    coefs = {}
    for k in cands:
        y = La[ara, Oa[:, k]] - La[ara, Oa[:, 1]]
        near = np.abs(va[:, k] - va[:, 1]) < 0.1 * sca
        for c in np.unique(ca):
            s = (ca == c) & near
            if s.sum() < 30: continue
            A = Fa[s]
            coefs[k, c] = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ y[s])

    def predict(X):
        o = aug_order(X); O = np.maximum(o, 0); cb = cfg_id(o); F = fn(X)
        pred = np.full((len(X), 3), -1e9); pred[:, 1] = 0.0  # S is the reference (0)
        for (k, c), co in coefs.items():
            t = cb == c
            pred[t, k] = F[t] @ co
        return O[np.arange(len(X)), pred.argmax(1)]

    return predict, sum(v.size for v in coefs.values())


if __name__ == "__main__":
    Xa, La, ma = data(81, 2_000_000)
    Xb, Lb, mb = data(82, 1_000_000)
    Ob = np.maximum(aug_order(Xb), 0)
    err = mb != np.argsort(Xb, 1)[:, -2]
    res = {}

    def score(pick, name):
        res[name] = dict(all=round(float((pick == mb).mean()), 5), err_set=round(float((pick[err] == mb[err]).mean()), 4))
        print(name, res[name], flush=True)

    hit = Ob == mb[:, None]
    for K in (2, 3, 4):
        res[f"oracle aug-ranks 1..{K-1} (no max)"] = round(float(hit[:, 1:K].any(1).mean()), 5)
        res[f"oracle aug-ranks 0..{K-1} (with max)"] = round(float(hit[:, :K].any(1).mean()), 5)
    print(json.dumps(res, indent=1))
    score(Ob[:, 1], "C1")
    for bank in BANKS:
        for cands in ((2,), (0, 2)):
            predict, ncoef = fit_percfg(Xa, La, bank, cands)
            score(predict(Xb), f"per-config linear, cands S+{'/'.join('MST'[k] for k in cands)}, bank [{bank}], {ncoef} coefs")
    json.dump(res, open("residual_percfg.json", "w"), indent=1)
