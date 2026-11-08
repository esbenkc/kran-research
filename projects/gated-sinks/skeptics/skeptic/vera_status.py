"""Vera: own implementation of PREREG section 3 (law fit + status) from per-model (sink, recovery) pairs.

Input: list of dicts {run, arm, gated(bool), sink, recovery}. No code from build is used.
"""
import numpy as np
from scipy import stats


def law_status(models, bar=0.05):
    ng = [m for m in models if not m["gated"]]
    g = [m for m in models if m["gated"]]
    x = np.array([m["sink"] for m in ng], float)
    y = np.array([m["recovery"] for m in ng], float)
    n = x.size
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - y.mean())).sum() / sxx
    a = y.mean() - b * xm
    res = y - (a + b * x)
    s_tot = np.sqrt((res**2).sum() / (n - 2))
    se_b = s_tot / np.sqrt(sxx)
    t = stats.t.ppf(0.975, n - 2)
    b_lo, b_hi = b - t * se_b, b + t * se_b
    R = x.max() - x.min()
    swing = abs(b) * R
    swing_hi = max(abs(b_lo), abs(b_hi)) * R
    rows = []
    for m in g:
        pred = a + b * m["sink"]
        h = t * s_tot * np.sqrt(1 / n + (m["sink"] - xm) ** 2 / sxx)
        band = max(0.03, 2 * s_tot, h)
        rows.append(dict(run=m["run"], sink=m["sink"], recovery=m["recovery"], pred=pred, resid=m["recovery"] - pred,
                         band=band, h=h, in_band=bool(abs(m["recovery"] - pred) <= band)))
    n_in = sum(r["in_band"] for r in rows)
    n_out = len(rows) - n_in
    rej_i = swing_hi < bar
    rej_ii = n_out >= 2
    rej_iii = b_lo > 0
    conf = b_hi < 0 and n_in >= 2
    status = "REJECTED" if (rej_i or rej_ii or rej_iii) else "CONFIRMED" if conf else "UNCONFIRMED"
    # sink levels: clusters of sorted non-gated sinks with neighbour gap > 0.1
    xs = np.sort(x)
    levels = 1 + int((np.diff(xs) > 0.1).sum())
    return dict(n=n, a=a, b=b, b_ci=[b_lo, b_hi], s_tot=s_tot, R=R, swing=swing, swing_hi=swing_hi,
                gated=rows, n_in=n_in, rej_i=bool(rej_i), rej_ii=bool(rej_ii), rej_iii=bool(rej_iii),
                conf=bool(conf), status=status, sink_levels=levels)
