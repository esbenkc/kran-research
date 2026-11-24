"""Pre-data operating characteristics of the frozen status rule (Vera, skeptic).

Rule simulated (Felix #29330 + Kasper #29332):
  per seed, per checkpoint: n balanced held-out prompts, Wilson 95%.
  seed_yes  = 2 consecutive task-held checkpoints (or the final one) with reader upper <= 0.60
  seed_no   = every task-held checkpoint has reader lower > 0.60   (length check assumed met)
  YES = >= 2 of 3 seeds seed_yes; NO = >= 2 of 3 seeds seed_no (symmetric); else UNCONFIRMED.
Task is assumed held at every checkpoint (worst case for multiple looks).
Usage: python3 vera_power.py [K checkpoints] [n per checkpoint]
"""
import sys
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
Z = 1.959963984540054
rng = np.random.default_rng(0)


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def status(p_true, sims=20000):
    k = rng.binomial(N, p_true, size=(sims, 3, K))
    lo, hi = wilson(k, N)
    low = hi <= 0.60
    consec = (low[:, :, 1:] & low[:, :, :-1]).any(-1) | low[:, :, -1]
    seed_no = (lo > 0.60).all(-1)
    yes = consec.sum(-1) >= 2
    no = seed_no.sum(-1) >= 2
    return yes.mean(), no.mean(), 1 - yes.mean() - no.mean()


if __name__ == "__main__":
    print(f"K={K} checkpoints/seed, n={N} prompts/checkpoint, 3 seeds, 20k sims")
    print(" true reader acc   P(YES)   P(NO)   P(UNCONF)")
    for p in [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80, 0.90]:
        y, n, u = status(p)
        print(f"      {p:.2f}        {y:.4f}  {n:.4f}  {u:.4f}")
