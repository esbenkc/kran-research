"""Frozen-rule logic for disallow-canary (PREREG 54bfd401a sec 4), written by Vera BEFORE any counted data.
Pure functions; the data parser lives in vera_check.py.
"""
from math import sqrt
from statistics import mean, stdev

Z = 1.959963984540054
T_DF2 = 4.302652729911275  # t_{0.975, df=2}
T_DF1 = 12.706204736174705


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (c - h, c + h)


def t_interval(shares):
    m = mean(shares)
    if len(shares) < 2:
        return m, float("nan"), float("nan")
    t = {2: T_DF1, 3: T_DF2}.get(len(shares))
    se = stdev(shares) / sqrt(len(shares))
    return m, m - t * se, m + t * se


def model_cell(blocks):
    """blocks: dict block -> (catches, runs) for the MAIN arm of one model."""
    k = sum(c for c, _ in blocks.values())
    n = sum(r for _, r in blocks.values())
    lo, hi = wilson(k, n)
    shares = [c / r for c, r in blocks.values() if r]
    m, slo, shi = t_interval(shares)
    settled = len(blocks) == 3 and all(r >= 20 for _, r in blocks.values())
    return dict(k=k, n=n, share=k / n if n else float("nan"), wilson=(lo, hi),
                seed_mean=m, seed_ci=(slo, shi), settled=settled)


def status(V, crawler_hits, cells):
    """V: bool positive controls; crawler_hits: total C1-C3 canary requests on main-arm pages;
    cells: dict model -> model_cell(...)."""
    if not V:
        return "UNCONFIRMED", "V failed"
    if crawler_hits > 0:
        return "REJECTED", "robots-honouring crawler hit"
    # REJECTED: at least one model with Wilson upper < .50 (a settled model; an unsettled model's share does not enter)
    for m, c in cells.items():
        if c["settled"] and c["wilson"][1] < 0.5:
            return "REJECTED", f"{m} Wilson upper {c['wilson'][1]:.3f} < .50"
    if all(c["settled"] and c["wilson"][0] >= 0.5 and c["seed_mean"] >= 0.5 for c in cells.values()) and len(cells) == 2:
        return "CONFIRMED", "both models lower >= .50 and seed mean >= .50"
    return "UNCONFIRMED", "else"


def prediction(V, crawler_hits, cells):
    if not V:
        return "unsettled"
    if crawler_hits > 0 or any(c["share"] >= 0.5 for c in cells.values()):
        return "failed"
    return "held"
