"""Iris: size of the frozen-bin chi-square under a TRUE geometric, df=2 vs df=3 (decision-based expected counts)."""
import numpy as np
from scipy import stats
CAP = 80
BINS = [(1, 3), (4, 10), (11, 30), (31, 80)]
def one(h, n, rng):
    n_at = np.zeros(CAP + 2); giv = np.zeros(CAP + 2)
    for _ in range(n):
        k = rng.geometric(h)
        if k <= CAP:
            n_at[1:k + 1] += 1; giv[k] += 1
        else:
            n_at[1:CAP] += 1  # censored at page 80: decisions 1..79
    G = giv.sum(); D = n_at.sum(); hh = G / D
    if G == 0: return None
    O = np.array([giv[a:b + 1].sum() for a, b in BINS]); N = np.array([n_at[a:b + 1].sum() for a, b in BINS])
    E = hh * N; keep = N > 0
    chi_v = (((O - E) ** 2)[keep] / E[keep]).sum()                       # Vera's statistic
    chi_p = (((O - E) ** 2)[keep] / (E[keep] * (1 - hh))).sum()          # full 2xB Pearson
    B = keep.sum()
    return [1 - stats.chi2.cdf(chi_v, 2) > .05, 1 - stats.chi2.cdf(chi_v, B - 1) > .05, 1 - stats.chi2.cdf(chi_p, B - 1) > .05]
for h in (0.01, 0.02, 0.05):
    for n in (40, 60):
        rng = np.random.default_rng(1); r = [one(h, n, rng) for _ in range(4000)]; r = np.array([x for x in r if x])
        print(f"h={h} n={n}: P(p>.05) df2={r[:,0].mean():.3f}  df=B-1={r[:,1].mean():.3f}  pearson2xB df=B-1={r[:,2].mean():.3f}")
