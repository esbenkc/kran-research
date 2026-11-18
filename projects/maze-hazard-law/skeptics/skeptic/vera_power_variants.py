"""Vera: false-REJECT and true-REJECT per agent under rule variants of #25718.
v0 = as written (testable & not holds -> fails)
v1 = KM median not reached in a testable agent -> untested, not a fail
v2 = v1 + a 2x miss counts as a fail only if the exact 95% Poisson interval of pilot give-ups also misses [0.5,2]x; else untested"""
import math, sys, numpy as np
from scipy import stats
from vera_power_frozen import sim_run, CAP, BINS, AGENTS

def pred_of(h3): return CAP if h3 <= 0 else min(math.log(0.5) / math.log(1 - min(h3, 0.999)), CAP)

def verdicts(runs):
    n_at = np.zeros(CAP + 2); giv = np.zeros(CAP + 2)
    for k, c in runs:
        n_at[1:k + 1] += 1
        if c == "g": giv[k] += 1
    G = giv.sum(); lost = sum(c == "l" for _, c in runs)
    e3 = giv[1:4].sum(); d3 = n_at[1:4].sum(); h3 = e3 / d3
    testable = e3 >= 1 and G >= 10 and G >= 0.5 * (G + lost)
    h = G / n_at[1:CAP + 1].sum()
    O = np.array([giv[a:b + 1].sum() for a, b in BINS]); E = np.array([h * n_at[a:b + 1].sum() for a, b in BINS])
    p = 1 - stats.chi2.cdf((((O - E) ** 2)[E > 0] / E[E > 0]).sum(), 2)
    S, med = 1.0, None
    for k in range(1, CAP + 1):
        if n_at[k]: S *= 1 - giv[k] / n_at[k]
        if med is None and S <= .5: med = k
    inside = lambda pr: med is not None and .5 <= pr / med <= 2
    two = inside(pred_of(h3))
    lo = stats.chi2.ppf(.025, 2 * e3) / 2 / d3 if e3 > 0 else 0.0
    hi = stats.chi2.ppf(.975, 2 * e3 + 2) / 2 / d3
    # does any hazard in [lo, hi] give a prediction inside the band? pred is monotone in h3
    some_inside = med is not None and pred_of(hi) <= 2 * med and pred_of(lo) >= .5 * med
    if not testable: return ("U", "U", "U")
    v0 = "H" if (p > .05 and med is not None and two) else "F"
    if p <= .05: return (v0, "F", "F")
    if med is None: return (v0, "U", "U")
    v1 = "H" if two else "F"
    v2 = "H" if two else ("U" if some_inside else "F")
    return (v0, v1, v2)

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
print("agent | n | P(fail) v0 v1 v2 | P(holds) v0 v1 v2")
for name, g in AGENTS.items():
    if g is None: continue
    for n in (40, 60):
        rng = np.random.default_rng(5); F = np.zeros(3); H = np.zeros(3)
        for _ in range(reps):
            v = verdicts([sim_run(g, 0.005, rng) for _ in range(n)])
            F += [x == "F" for x in v]; H += [x == "H" for x in v]
        print(name, n, (F / reps).round(3).tolist(), (H / reps).round(3).tolist())
