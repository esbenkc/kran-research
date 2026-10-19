"""Vera's own checker core (skeptic). Written from PREREG.md s6-8 only; does not import build's code.

AUROC via the Mann-Whitney rank formula (ties = 0.5), my own class-balanced draws, my own paired bootstrap.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count half. y < 0 (OTHER) is dropped."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    keep = y >= 0
    y, s = y[keep].astype(bool), s[keep]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def balanced_draw(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray | None:
    """n/2 positives + n/2 negatives, without replacement. None if a class is too small."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    h = n // 2
    if len(pos) < h or len(neg) < h:
        return None
    return np.concatenate([rng.choice(pos, h, replace=False), rng.choice(neg, h, replace=False)])


def fit_lr(X: np.ndarray, y: np.ndarray, C: float = 0.05):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)
    return sc, lr


def score(model, X: np.ndarray) -> np.ndarray:
    sc, lr = model
    return lr.decision_function(sc.transform(X))


def catch_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.05) -> float:
    """Share of positives scoring above the threshold that flags at most `fpr` of negatives (threshold on negatives)."""
    y = np.asarray(y).astype(bool)
    neg = np.sort(s[~y])[::-1]
    k = int(np.floor(fpr * len(neg)))  # number of negatives allowed above the threshold
    thr = neg[k] if k < len(neg) else -np.inf  # flag strictly above the (k+1)-th highest negative
    return float((s[y] > thr).mean())


def paired_boot_gap(y_a, s_a, y_b, s_b, n_boot=1000, seed=1019):
    """Gap AUROC(a) - AUROC(b) where a and b are scored on the SAME prompts (index-aligned). Resample prompts."""
    rng = np.random.default_rng(seed)
    m = len(y_a)
    assert len(y_b) == m
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, m, m)
        out[i] = auroc(y_a[idx], s_a[idx]) - auroc(y_b[idx], s_b[idx])
    return float(auroc(y_a, s_a) - auroc(y_b, s_b)), np.nanpercentile(out, [2.5, 97.5])
