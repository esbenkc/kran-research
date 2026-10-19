"""Vera's independent checker core (theory-1019-max-pool-survives). numpy + sklearn only; no build code.

Definitions taken from PREREG.md (commit 9bc1a08 + Amendment 1 at 4bc18f2), not from analysis.py:
  positive = FALSE statement; token 0 already excluded in the saved token arrays (tok_pos starts at 1).
  threshold = 2nd-highest of the 200 refit TRUE scores of the scored model state; flag iff s > t.
  catch = share of the 596 eval FALSE statements flagged.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

_H = Path(__file__).resolve().parents[2]
# works from swarm-<room>/skeptic/ and from kran-research/projects/max-pool-survives/skeptics/skeptic/
PROJ = next(p for p in (_H / "kran-research/projects/max-pool-survives", _H) if (p / "PREREG.md").exists())
DATA = PROJ / "data"
CGRID = (0.01, 0.1, 1.0, 10.0)


def meta():
    s = json.loads((DATA / "split.json").read_text())
    y = np.array([1 - r["true"] for r in s])                     # 1 = false statement
    split = np.array([r["split"] for r in s])
    city = np.array([r["city"] for r in s])
    src = np.array([r["src"] for r in s])
    pair = np.array([f"{r['city']}|{r['src']}" for r in s])      # true/false twins share the prefix
    return dict(y=y, split=split, city=city, src=src, pair=pair, text=[r["text"] for r in s])


def thr(neg, k=2):
    return np.sort(neg)[-k]


def auroc(neg, pos):
    allv = np.r_[neg, pos]
    r = np.empty(len(allv))
    order = allv.argsort(kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def seg_reduce(v, off, how):
    """v: per-token values (T,) or (T,d); off: (S+1,) offsets. Returns per-statement reduction."""
    if how == "mean":
        cs = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, 0, dtype=np.float64)])
        return (cs[off[1:]] - cs[off[:-1]]) / (off[1:] - off[:-1]).reshape((-1,) + (1,) * (v.ndim - 1))
    if how == "max":
        return np.maximum.reduceat(v, off[:-1], axis=0)
    if how == "last":
        return v[off[1:] - 1]
    raise ValueError(how)


def tok_rows(off, stmts):
    """Token indices belonging to the given statement indices."""
    return np.concatenate([np.arange(off[i], off[i + 1]) for i in stmts])


class Lin:
    """Logistic regression on standardised features, mapped back to raw space: s = X @ w + b."""

    def __init__(self, X, y, C):
        mu, sd = X.mean(0), X.std(0) + 1e-8
        m = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, y)
        self.w = m.coef_[0] / sd
        self.b = float(m.intercept_[0] - (mu / sd) @ m.coef_[0])

    def __call__(self, X):
        return X @ self.w + self.b


def fit_pooled(Xs, y, groups, readout_fn, fit_rows_fn, Cgrid=CGRID, seed=0):
    """Generic monitor fit with C chosen by 5-fold GroupKFold on the AUROC of the monitor's own pooled score.
    Xs: per-statement feature source; fit_rows_fn(stmt_idx) -> (Xfit, yfit); readout_fn(lin, stmt_idx) -> scores."""
    idx = np.arange(len(y))
    gkf = GroupKFold(n_splits=5)
    cv = {}
    for C in Cgrid:
        a = []
        for tr, te in gkf.split(idx, y, groups):
            Xf, yf = fit_rows_fn(idx[tr])
            lin = Lin(Xf, yf, C)
            sc = readout_fn(lin, idx[te])
            a.append(auroc(sc[y[te] == 0], sc[y[te] == 1]))
        cv[C] = float(np.mean(a))
    best = max(cv.values())
    Cbest = min(C for C in Cgrid if cv[C] >= best - 1e-12)     # ties -> smaller C
    Xf, yf = fit_rows_fn(idx)
    return Lin(Xf, yf, Cbest), Cbest, cv
