"""Vera's metric core: only numpy + the PREREG definitions. No build code is used."""
import numpy as np
from scipy.stats import norm


def thr(neg, k=2):
    """Flag iff s > k-th largest of the negatives (k=2 -> 2/201 = 1.0% fresh FPR)."""
    return np.sort(neg)[-k]


def auroc(neg, pos):
    """Mann-Whitney AUROC with ties counted 1/2."""
    allv = np.r_[neg, pos]
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    # average ranks for ties
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[len(neg):].sum()
    n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n0 * n1)


def decomp(clean, att, k=2):
    """clean/att: dict with 'refit_neg', 'eval_pos', 'eval_neg' score arrays (same rows, same order).
    Returns lost, recovered, share, plus TPRs and realised FPRs."""
    t0 = thr(clean["refit_neg"], k)
    t1 = thr(att["refit_neg"], k)
    tc = (clean["eval_pos"] > t0).mean()
    told = (att["eval_pos"] > t0).mean()
    tref = (att["eval_pos"] > t1).mean()
    lost, rec = tc - told, tref - told
    return dict(t0=t0, t1=t1, tpr_clean=tc, tpr_old=told, tpr_refit=tref, lost=lost, recovered=rec,
                share=rec / lost if lost > 0 else np.nan,
                fpr_t0_clean=(clean["eval_neg"] > t0).mean(), fpr_t0_att=(att["eval_neg"] > t0).mean(),
                fpr_t1_att=(att["eval_neg"] > t1).mean(),
                auc_clean=auroc(clean["eval_neg"], clean["eval_pos"]),
                auc_att=auroc(att["eval_neg"], att["eval_pos"]))


def band(att, clean):
    """Two-sided band refit on attacked refit-200: flag iff s > max or s < min (fresh FPR 2/201)."""
    lo, hi = att["refit_neg"].min(), att["refit_neg"].max()
    return ((att["eval_pos"] > hi) | (att["eval_pos"] < lo)).mean(), ((att["eval_neg"] > hi) | (att["eval_neg"] < lo)).mean()


def share_formula(D, c, p, z=norm.ppf(1 - 2 / 201)):
    return (norm.cdf(D - (1 - c) * p - z) - norm.cdf(D - p - z)) / (norm.cdf(D - z) - norm.cdf(D - p - z))


def paired_boot(cleans, atts, n=2000, k=2, seed=0):
    """cleans/atts: lists (one per run) of dicts. One resample of refit-200 and eval-pos indices per draw,
    applied to all runs. Returns array of pooled shares."""
    rng = np.random.default_rng(seed)
    nr, npos = len(cleans[0]["refit_neg"]), len(cleans[0]["eval_pos"])
    out = np.empty(n)
    for d in range(n):
        ir, ip = rng.integers(0, nr, nr), rng.integers(0, npos, npos)
        L = R = 0.0
        for c, a in zip(cleans, atts):
            t0, t1 = thr(c["refit_neg"][ir], k), thr(a["refit_neg"][ir], k)
            cp, ap = c["eval_pos"][ip], a["eval_pos"][ip]
            told = (ap > t0).mean()
            L += (cp > t0).mean() - told
            R += (ap > t1).mean() - told
        out[d] = R / L
    return out


def indep_boot(cleans, atts, n=2000, k=2, seed=0):
    """Independent per-run resampling (the rejected PREREG draft), for comparison only."""
    rng = np.random.default_rng(seed)
    nr, npos = len(cleans[0]["refit_neg"]), len(cleans[0]["eval_pos"])
    out = np.empty(n)
    for d in range(n):
        L = R = 0.0
        for c, a in zip(cleans, atts):
            ir, ip = rng.integers(0, nr, nr), rng.integers(0, npos, npos)
            t0, t1 = thr(c["refit_neg"][ir], k), thr(a["refit_neg"][ir], k)
            cp, ap = c["eval_pos"][ip], a["eval_pos"][ip]
            told = (ap > t0).mean()
            L += (cp > t0).mean() - told
            R += (ap > t1).mean() - told
        out[d] = R / L
    return out
