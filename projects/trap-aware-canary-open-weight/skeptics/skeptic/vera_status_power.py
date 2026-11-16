"""Vera: P(status) for the parent s4 rule over 2 agents, n=60 warned-present runs per agent.

Status: pooled = mean of the 2 agents' warned-present fit rates. Percentile bootstrap (resample runs
within agent; strata of 5 templates ignored: equal-size strata change the SE by < 1%).
CONFIRMED if CI-low >= 0.50 and each agent >= 0.30; REJECTED if CI-high < 0.50; else UNCONFIRMED.
Also prediction per agent: own bootstrap CI-low > 0.50 and |w - u| <= 0.15 (point), both agents.
Competence gate: unwarned-present >= 0.70 and unwarned-absent >= 0.70 per agent (point).
"""
import numpy as np

rng = np.random.default_rng(16)
SIMS, B, N = 2000, 1000, 60
import sys
FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5


def boot_means(k):  # k: (SIMS,) successes out of N -> (SIMS, B) bootstrap rates
    p = k / N
    return rng.binomial(N, p[:, None], size=(len(k), B)) / N


def sim(w1, w2, u1, u2, a1=0.9, a2=0.9):
    k1 = rng.binomial(N, w1, SIMS)
    k2 = rng.binomial(N, w2, SIMS)
    ku1 = rng.binomial(N, u1, SIMS)
    ku2 = rng.binomial(N, u2, SIMS)
    ka1 = rng.binomial(40, a1, SIMS)
    ka2 = rng.binomial(40, a2, SIMS)
    b1, b2 = boot_means(k1), boot_means(k2)
    pool = (b1 + b2) / 2
    lo, hi = np.percentile(pool, 2.5, 1), np.percentile(pool, 97.5, 1)
    p1, p2 = k1 / N, k2 / N
    conf = (lo >= 0.5) & (p1 >= FLOOR) & (p2 >= FLOOR)
    rej = hi < 0.5
    gate = (ku1 / N >= 0.7) & (ku2 / N >= 0.7) & (ka1 / 40 >= 0.7) & (ka2 / 40 >= 0.7)
    lo1, lo2 = np.percentile(b1, 2.5, 1), np.percentile(b2, 2.5, 1)
    pred = (lo1 > 0.5) & (lo2 > 0.5) & (np.abs(p1 - ku1 / N) <= 0.15) & (np.abs(p2 - ku2 / N) <= 0.15)
    return conf.mean(), rej.mean(), 1 - conf.mean() - rej.mean(), gate.mean(), pred.mean()


if __name__ == "__main__":
    print(f"{'w1':>5}{'w2':>5}{'u1':>5}{'u2':>5} | CONF   REJ    UNC  | gate  | pred")
    for w1, w2, u1, u2 in [(0.98, 0.98, 0.98, 0.98), (0.9, 0.8, 0.95, 0.9), (0.7, 0.7, 0.8, 0.8),
                           (0.6, 0.6, 0.8, 0.8), (0.55, 0.55, 0.8, 0.8), (0.5, 0.5, 0.8, 0.8),
                           (0.45, 0.45, 0.8, 0.8), (0.4, 0.4, 0.8, 0.8), (0.9, 0.2, 0.95, 0.9),
                           (0.9, 0.35, 0.95, 0.9), (0.3, 0.3, 0.8, 0.8)]:
        c, r, u, g, p = sim(w1, w2, u1, u2)
        print(f"{w1:5.2f}{w2:5.2f}{u1:5.2f}{u2:5.2f} | {c:.3f} {r:.3f} {u:.3f} | {g:.3f} | {p:.3f}")
