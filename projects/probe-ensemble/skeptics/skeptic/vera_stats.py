"""Vera: shared stats helpers (own AUROC with ties, FPR threshold)."""
import numpy as np


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def thr_at_fpr(s, y, fpr=0.05):
    neg = np.sort(np.asarray(s)[y == 0])
    return np.quantile(neg, 1 - fpr)
