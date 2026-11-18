"""Vera: pass rates of Felix's FINAL PLAN v1 frozen rule (#25718) on simulated agents, before data.

Per run, page k = 1..80 (one new maze page per call, an upper bound on pages):
  give-up with prob g(k), else lost with prob l (censored at k), else go on; reaching 80 = censored.
Rule:
  h3 = give-ups on pages 1-3 / decisions on pages 1-3
  TESTABLE = >=1 pilot give-up AND >=10 give-ups before cap AND give-ups >= half of all exits
  chi-square: geometric h = give-ups / decisions (all pages); bins 1-3, 4-10, 11-30, 31-80;
    expected give-ups in a bin = h * decisions in the bin; df = 4 - 1 - 1 = 2
  KM on give-ups (lost and cap censored); median reached iff S(80) <= 0.5
  pred = min(ln0.5/ln(1-h3), 80); HOLDS = TESTABLE and p > .05 and median reached and pred/KMmed in [0.5, 2]
"""
import math
import sys
import numpy as np
from scipy import stats

CAP = 80
BINS = [(1, 3), (4, 10), (11, 30), (31, 80)]


def sim_run(g, l, rng):
    for k in range(1, CAP + 1):
        u = rng.random()
        if u < g(k):
            return k, "g"
        if u < g(k) + l:
            return k, "l"
    return CAP, "c"


def analyse(runs):
    n_at = np.zeros(CAP + 2)  # decisions made on page k
    giv = np.zeros(CAP + 2)
    for k, cause in runs:
        n_at[1:k + 1] += 1
        if cause == "g":
            giv[k] += 1
    G = giv.sum()
    lost = sum(1 for _, c in runs if c == "l")
    pilot_g = giv[1:4].sum()
    h3 = pilot_g / n_at[1:4].sum()
    testable = pilot_g >= 1 and G >= 10 and G >= 0.5 * (G + lost)
    h = G / n_at[1:CAP + 1].sum()
    O = np.array([giv[a:b + 1].sum() for a, b in BINS])
    E = np.array([h * n_at[a:b + 1].sum() for a, b in BINS])
    keep = E > 0
    chi = (((O - E) ** 2)[keep] / E[keep]).sum()
    p = 1 - stats.chi2.cdf(chi, 2)
    S, med = 1.0, None
    for k in range(1, CAP + 1):
        if n_at[k] > 0:
            S *= 1 - giv[k] / n_at[k]
        if med is None and S <= 0.5:
            med = k
    pred = CAP if h3 == 0 else min(math.log(0.5) / math.log(1 - h3), CAP)
    two = med is not None and 0.5 <= pred / med <= 2
    holds = testable and p > 0.05 and med is not None and two
    cap_share = sum(1 for _, c in runs if c == "c") / len(runs)
    return dict(testable=testable, chi_ok=p > 0.05, med=med is not None, two=two, holds=holds,
                cap50=cap_share >= 0.5, rejects=testable and not holds)


def rate(g, l, n, reps, seed):
    rng = np.random.default_rng(seed)
    keys = ["testable", "chi_ok", "med", "two", "holds", "rejects", "cap50"]
    acc = dict.fromkeys(keys, 0)
    for _ in range(reps):
        r = analyse([sim_run(g, l, rng) for _ in range(n)])
        for key in keys:
            acc[key] += r[key]
    return {key: round(v / reps, 3) for key, v in acc.items()}


AGENTS = {
    "geo h=.005": lambda k: 0.005,
    "geo h=.01": lambda k: 0.01,
    "geo h=.02": lambda k: 0.02,
    "geo h=.05": lambda k: 0.05,
    "falling (Weibull k=.6, med~20)": lambda k: 1 - math.exp(-((k / 36) ** 0.6) + (((k - 1) / 36) ** 0.6)),
    "rising (Weibull k=1.6, med~40)": lambda k: 1 - math.exp(-((k / 50) ** 1.6) + (((k - 1) / 50) ** 1.6)),
    "burst: .01 then 30% at page 40": lambda k: 0.3 if k == 40 else 0.01,
    "mix 30% fast(.3) / 70% stay(.005)": None,  # handled below
}

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print("agent | lost/page | n | testable chi_ok med_reached 2x HOLDS REJECTS cap>=50%")
    for name, g in AGENTS.items():
        for l in (0.0, 0.01):
            for n in (40, 60):
                if g is None:  # per-run mixture: draw the run type, then simulate
                    rng = np.random.default_rng(7)
                    tot = dict.fromkeys(["testable", "chi_ok", "med", "two", "holds", "rejects", "cap50"], 0)
                    for _ in range(reps):
                        runs = [sim_run((lambda k: 0.3) if rng.random() < 0.3 else (lambda k: 0.005), l, rng) for _ in range(n)]
                        r = analyse(runs)
                        for key in tot:
                            tot[key] += r[key]
                    res = {key: round(v / reps, 3) for key, v in tot.items()}
                else:
                    res = rate(g, l, n, reps, 11)
                print(f"{name} | {l} | {n} | " + " ".join(str(res[k]) for k in ["testable", "chi_ok", "med", "two", "holds", "rejects", "cap50"]))
