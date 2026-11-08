"""Vera: simulate Tove's amended status rule (#16854) before data.
Statistic: d = mean over agents of (fit rate - nonfit rate), stuck title cell, exposed runs.
CI: run-level bootstrap (resample runs within each agent x anchor cell), 95% percentile.
YES (checked first): d >= 0.05 AND CI-low > 0.  NO: CI-high < 0.15.  else NOT SETTLED.
"""
import sys
import numpy as np

rng = np.random.default_rng(11082)
B = 1000


def status_batch(f, o, n):
    # f, o: (reps, agents) counts out of n
    pf, po = f / n, o / n
    d = (pf - po).mean(1)
    bf = rng.binomial(n, pf[:, None, :].repeat(B, 1)) / n  # (reps, B, agents)
    bo = rng.binomial(n, po[:, None, :].repeat(B, 1)) / n
    bd = (bf - bo).mean(2)
    lo = np.quantile(bd, 0.025, axis=1)
    hi = np.quantile(bd, 0.975, axis=1)
    yes = (d >= 0.05) & (lo > 0)
    no = (~yes) & (hi < 0.15)
    return yes, no


def sim(base, gap, n, agents, reps=2000):
    pn = np.full(agents, base)
    pf = np.clip(pn + gap, 0, 1)
    f = rng.binomial(n, pf, size=(reps, agents))
    o = rng.binomial(n, pn, size=(reps, agents))
    yes, no = status_batch(f, o, n)
    return yes.mean(), no.mean(), 1 - yes.mean() - no.mean()


if __name__ == "__main__":
    for agents in (3, 4):
        for n in (40, 60):
            for base in (0.1, 0.3, 0.7):
                for gap in (0.0, 0.15, 0.30):
                    y, no, ns = sim(base, gap, n, agents)
                    print(f"agents={agents} n={n} nonfit={base:.1f} gap={gap:+.2f}: "
                          f"YES {y:.3f} NO {no:.3f} NOT-SETTLED {ns:.3f}", flush=True)
