"""Vera: simulate Felix's draft status rule before data.
Rule: pooled (mean over agents) fit-minus-nonfit follow rate, exposed text-view runs,
no-visible-route setting. 95% CI low > 0 -> YES; CI high < +0.10 -> NO; else NOT SETTLED.
CI: normal approx on mean of per-agent differences (agents fixed), and a run-level bootstrap.
"""
import numpy as np

rng = np.random.default_rng(1108)


def status(fit, non, n):
    # fit, non: arrays (agents,) of success counts out of n per cell, per view-pooled
    pf, pn = fit / n, non / n
    d = (pf - pn).mean()
    var = ((pf * (1 - pf) + pn * (1 - pn)) / n).sum() / len(pf) ** 2
    se = np.sqrt(max(var, 1e-12))
    lo, hi = d - 1.96 * se, d + 1.96 * se
    if lo > 0:
        return "YES"
    if hi < 0.10:
        return "NO"
    return "NS"


def sim(base_non, gap, n, agents=4, agent_sd=0.0, reps=4000):
    out = {"YES": 0, "NO": 0, "NS": 0}
    for _ in range(reps):
        a = rng.normal(0, agent_sd, agents)
        pn = np.clip(base_non + a, 0, 1)
        pf = np.clip(base_non + gap + a, 0, 1)
        f = rng.binomial(n, pf)
        o = rng.binomial(n, pn)
        out[status(f, o, n)] += 1
    return {k: round(v / reps, 3) for k, v in out.items()}


if __name__ == "__main__":
    print("n = runs per agent per anchor cell (after pooling views), 4 agents")
    for n in (20, 30, 40, 60):
        for base in (0.0, 0.05, 0.3, 0.7):
            for gap in (0.0, 0.05, 0.10, 0.15, 0.30):
                if base + gap > 1:
                    continue
                r = sim(base, gap, n)
                print(f"n={n:3d} nonfit={base:.2f} gap={gap:.2f} -> {r}")
