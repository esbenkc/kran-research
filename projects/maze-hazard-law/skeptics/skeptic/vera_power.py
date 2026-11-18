"""Vera: pass rates of the scout's frozen rule on simulated agents (before data).

Rule as read from the goal (refined when PREREG.md is frozen):
  per agent, N runs, cap = 80 pages (one new page per call, an upper bound).
  h3   = exits on pages 1-3 / page-visits at risk on pages 1-3
  pred = geometric median pages = ceil(ln 0.5 / ln(1 - h3)) (inf if h3 == 0)
  obs  = median pages per run, capped runs count as the cap
  A = chi-square of a censored geometric MLE fit (bins 1-3, 4-10, 11-30, 31-79, cap) p > 0.05
  B = pred / obs in [0.5, 2]
  C = share of runs at the cap >= 0.5 (pooled over both agents in the real rule; here one agent)
  holds = A and B and C
Variant 'capped': pred is truncated at the cap before the 2x test (the vacuous reading).
"""
import math
import numpy as np
from scipy import stats

CAP = 80
BINS = [(1, 3), (4, 10), (11, 30), (31, 79)]


def sim(kind, n, rng):
    out = []
    for _ in range(n):
        if kind[0] == "geo":
            h = kind[1]
            k = rng.geometric(h) if h > 0 else CAP
        elif kind[0] == "burst":  # tiny hazard, then a burst of quits at a budget fraction
            h, at, q = kind[1], kind[2], kind[3]
            k = rng.geometric(h)
            if k > at and rng.random() < q:
                k = at + rng.integers(0, 4)
        elif kind[0] == "mix":  # two kinds of run: fast quitters and stayers
            f, hf, hs = kind[1], kind[2], kind[3]
            k = rng.geometric(hf if rng.random() < f else hs)
        out.append(min(k, CAP))
    return np.array(out)  # pages seen; value CAP = censored at the cap


def fit_test(p):
    exits = p[p < CAP]
    # censored geometric MLE: exits / total page-visits
    visits = p.sum()
    h = len(exits) / visits if visits else 0.0
    obs, exp = [], []
    surv = lambda k: (1 - h) ** k  # P(pages > k)
    for a, b in BINS:
        obs.append(((p >= a) & (p <= b) & (p < CAP)).sum())
        exp.append(len(p) * (surv(a - 1) - surv(b)))
    obs.append((p == CAP).sum())
    exp.append(len(p) * surv(CAP - 1))
    obs, exp = np.array(obs, float), np.array(exp, float)
    if h == 0:
        return h, 1.0
    keep = exp > 0
    chi = (((obs - exp) ** 2) / np.where(keep, exp, 1))[keep].sum()
    df = keep.sum() - 1 - 1
    return h, 1 - stats.chi2.cdf(chi, max(df, 1))


def judge(p, capped):
    at_risk = sum(min(x, 3) for x in p)
    ex3 = ((p <= 3) & (p < CAP)).sum()
    h3 = ex3 / at_risk
    pred = math.inf if h3 == 0 else math.ceil(math.log(0.5) / math.log(1 - h3))
    if capped:
        pred = min(pred, CAP)
    obs = np.median(p)
    _, pval = fit_test(p)
    A = pval > 0.05
    B = 0.5 <= pred / obs <= 2 if math.isfinite(pred) else False
    C = (p == CAP).mean() >= 0.5
    return A, B, C


def rates(kind, n=40, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    res = {"strict": np.zeros(4), "capped": np.zeros(4)}
    for _ in range(reps):
        p = sim(kind, n, rng)
        for mode in res:
            A, B, C = judge(p, mode == "capped")
            res[mode] += [A, B, C, A and B and C]
    return {m: (v / reps).round(3).tolist() for m, v in res.items()}


if __name__ == "__main__":
    kinds = [("geo", 0.003), ("geo", 0.008), ("geo", 0.02), ("geo", 0.05), ("geo", 0.15),
             ("burst", 0.003, 40, 0.6), ("burst", 0.003, 20, 0.7),
             ("mix", 0.3, 0.5, 0.002), ("mix", 0.2, 0.5, 0.005)]
    print("kind | mode | P(A chi ok), P(B 2x), P(C >=50% cap), P(all)")
    for k in kinds:
        r = rates(k)
        for m, v in r.items():
            print(k, m, v)
