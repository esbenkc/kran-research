"""Pre-data sim of Felix's v2 draft FROZEN RULE (#32381), whole rule:
gate: bare catch >= 60% per model (2 x 20 runs);
CONFIRMED: both models Wilson lower >= .50 on 90 main runs AND seed-level mean (3 blocks x 30) >= .50;
REJECTED: both models Wilson upper < .50; else UNCONFIRMED (also if any gate fails).
Crawler preconditions treated as passed (deterministic).
"""
import numpy as np
from math import sqrt

Z = 1.959964
rng = np.random.default_rng(11282)


def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def one(p_main, p_bare, block_sd=0.0):
    status = []
    for _ in range(2):  # two models, same truth
        bare = rng.binomial(40, p_bare)
        if bare < 24:  # 60% of 40
            return "UNCONFIRMED"
        blocks = [rng.binomial(30, float(np.clip(p_main + rng.normal(0, block_sd), 0, 1))) for _ in range(3)]
        x = sum(blocks)
        lo, hi = wilson(x, 90)
        mean = np.mean([b / 30 for b in blocks])
        if lo >= 0.5 and mean >= 0.5:
            status.append("C")
        elif hi < 0.5:
            status.append("R")
        else:
            status.append("U")
    if status == ["C", "C"]:
        return "CONFIRMED"
    if status == ["R", "R"]:
        return "REJECTED"
    return "UNCONFIRMED"


print("true main p | P(CONF) P(REJ) P(UNC)   (bare .85, block sd 0 / .08)")
for sd in (0.0, 0.08):
    for p in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
        r = [one(p, 0.85, sd) for _ in range(8000)]
        print(f"sd {sd:.2f} p {p:.2f} | {r.count('CONFIRMED')/8000:.3f} {r.count('REJECTED')/8000:.3f} {r.count('UNCONFIRMED')/8000:.3f}")
print("bare-gate pass prob per model at bare p:", {pb: round(float(np.mean(rng.binomial(40, pb, 20000) >= 24)), 3) for pb in (0.6, 0.65, 0.7, 0.8)})
