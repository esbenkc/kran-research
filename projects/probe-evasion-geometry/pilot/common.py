"""Shared math for the probe-evasion geometry test.

Notation (matches the post):
  Sigma  within-class covariance of the probed representation
  delta  class-mean gap  mu_1 - mu_0
  d(u)   = u.delta / sqrt(u' Sigma u)          separation of probe u in noise units
  AUROC(u) = Phi(d(u) / sqrt(2))               (two Gaussians, shared covariance)
  cos_S(u, w) = u' Sigma w / sqrt(u' Sigma u * w' Sigma w)   whitened cosine
Lemma: the smallest move (in Sigma^{-1} norm) of the class-1 mean that removes kappa noise
units of separation from probe w is  Delta = -kappa * Sigma w / sqrt(w' Sigma w), and then
  d'(u) = d(u) - kappa * cos_S(u, w)   for every probe u.
"""
import numpy as np
from scipy.stats import norm


def auroc(scores, y):
    """Rank AUROC (Mann-Whitney), ties averaged."""
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(bool)
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def class_stats(X, y):
    y = np.asarray(y).astype(bool)
    mu0, mu1 = X[~y].mean(0), X[y].mean(0)
    Xc = np.concatenate([X[~y] - mu0, X[y] - mu1])
    Sigma = Xc.T @ Xc / (len(X) - 2)
    return mu0, mu1, Sigma


def sep(u, delta, Sigma):
    return float(u @ delta / np.sqrt(u @ Sigma @ u))


def cos_white(u, w, Sigma):
    return float(u @ Sigma @ w / np.sqrt((u @ Sigma @ u) * (w @ Sigma @ w)))


def cos_raw(u, w):
    return float(u @ w / np.linalg.norm(u) / np.linalg.norm(w))


def d_from_auroc(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.sqrt(2) * norm.ppf(a)


def auroc_from_d(d):
    return norm.cdf(np.asarray(d) / np.sqrt(2))


def lda(delta, Sigma, ridge=1e-6):
    return np.linalg.solve(Sigma + ridge * np.trace(Sigma) / len(Sigma) * np.eye(len(Sigma)), delta)


def probe_family(delta, Sigma, w, n, rng):
    """n valid probes at varied angles to w: LDA probes restricted to random subspaces
    of random dimension, half of which are tilted toward w so the whitened cosines span [0, 1]."""
    D = len(delta)
    probes = []
    while len(probes) < n:
        k = int(rng.integers(1, max(2, D // 2)))
        P = rng.standard_normal((D, k))
        if len(probes) % 2 == 1:  # tilt: include a noisy copy of w in the subspace
            P[:, 0] = w / np.linalg.norm(w) + rng.uniform(0.0, 2.0) * rng.standard_normal(D) / np.sqrt(D)
        Q, _ = np.linalg.qr(P)
        a = np.linalg.solve(Q.T @ Sigma @ Q, Q.T @ delta)
        u = Q @ a
        if sep(u, delta, Sigma) > 0.5:  # keep only probes that work (AUROC > ~0.64)
            probes.append(u / np.linalg.norm(u))
    return np.array(probes)


def r2_identity(meas, pred):
    meas, pred = np.asarray(meas), np.asarray(pred)
    return float(1 - ((meas - pred) ** 2).sum() / ((meas - meas.mean()) ** 2).sum())


def r2_corr(meas, pred):
    return float(np.corrcoef(meas, pred)[0, 1] ** 2)
