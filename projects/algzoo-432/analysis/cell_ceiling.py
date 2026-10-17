"""Iris: ceiling for compact near-tie rules (research.md s.6).

Run: .venv/bin/python cell_ceiling.py   (~1 min, ~3 GB RAM). Dev seed 21; never the holdout.

Rule family: per-cell majority vote of the model's choice (coded as rank class: max / 2nd / 3rd / other).
Cell = (pM, pS, pT) x quantile bins of (S-T)/|M| and (M-S)/|M| x number of positive among (S, T)
[x optional extra binned feature]. Train on the first 5e6 samples, test on the last 1e6.
This upper-bounds any rule that reads only those features (weight-derived or fitted).
"""
import numpy as np

from core import load_raw

Wi, Wh, Wo = load_raw()
N_ALL, N_TR = 6_000_000, 5_000_000


def model_pred(x):
    out = np.empty(len(x), np.int64)
    for i in range(0, len(x), 500_000):
        h = np.zeros((len(x[i : i + 500_000]), 16))
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[i : i + 500_000, t : t + 1] * Wi[None, :], 0.0)
        out[i : i + 500_000] = (h @ Wo.T).argmax(1)
    return out


X = np.random.default_rng(21).standard_normal((N_ALL, 10))
Y = model_pred(X)
order = np.argsort(-X, 1)
ar = np.arange(N_ALL)
pM, pS, pT, p4 = order[:, 0], order[:, 1], order[:, 2], order[:, 3]
M, S, T, U = X[ar, pM], X[ar, pS], X[ar, pT], X[ar, p4]
cls = np.where(Y == pS, 1, np.where(Y == pT, 2, np.where(Y == pM, 0, 3)))
tr, te = slice(0, N_TR), slice(N_TR, N_ALL)
g1, g2 = (M - S) / np.abs(M), (S - T) / np.abs(M)
sign = (S > 0).astype(int) + (T > 0).astype(int)


def qbin(v, nb):
    edges = np.quantile(v[tr], np.linspace(0, 1, nb + 1)[1:-1])
    return np.searchsorted(edges, v)


def ceiling(nb2, nb1, extra=None):
    cell = ((pM * 10 + pS) * 10 + pT).astype(np.int64)
    cell = (cell * nb2 + qbin(g2, nb2)) * nb1 + qbin(g1, nb1)
    cell = cell * 3 + sign
    if extra is not None:
        v, nb = extra
        cell = cell * nb + qbin(v, nb)
    u, ci = np.unique(cell, return_inverse=True)
    cnt = np.zeros((len(u), 4))
    np.add.at(cnt, (ci[tr], cls[tr]), 1)
    maj = cnt.argmax(1)
    maj[cnt.sum(1) == 0] = 1
    pred = maj[ci]
    return (pred[tr] == cls[tr]).mean(), (pred[te] == cls[te]).mean(), len(u)


def main():
    print("class shares (max, 2nd, 3rd, other):", np.round(np.bincount(cls, minlength=4) / N_ALL, 4))
    rows = [
        ("12 x 8 gap bins", ceiling(12, 8)),
        ("24 x 8", ceiling(24, 8)),
        ("40 x 12", ceiling(40, 12)),
        ("12 x 4 + x_9/|M| (4 bins)", ceiling(12, 4, (X[:, 9] / np.abs(M), 4))),
        ("12 x 4 + x_8/|M| (4 bins)", ceiling(12, 4, (X[:, 8] / np.abs(M), 4))),
        ("12 x 4 + (T - 4th)/|M| (4 bins)", ceiling(12, 4, ((T - U) / np.abs(M), 4))),
    ]
    for name, (a_tr, a_te, n) in rows:
        print(f"{name:34s} cells {n:7d}  train {a_tr:.4f}  test {a_te:.4f}")


if __name__ == "__main__":
    main()
