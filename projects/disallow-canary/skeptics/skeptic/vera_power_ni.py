"""Power of a non-inferiority clause 'lower 95% bound of (treated - bare) > -10 pts'.
Pre-data sim on parent rates only (nofollow-canary: 72-74/90, 14/90 discordant pairs).
Two interval methods: Newcombe hybrid score (unpaired) and paired Wald on discordant pairs.
"""
import numpy as np
from math import sqrt

Z = 1.959964
rng = np.random.default_rng(1128)


def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_lo(x1, n1, x0, n0):
    p1, p0 = x1 / n1, x0 / n0
    l1, u1 = wilson(x1, n1)
    l0, u0 = wilson(x0, n0)
    return (p1 - p0) - sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)


def paired_lo(b, c, n):
    # b = bare-only follows, c = treated-only follows; diff = (c-b)/n
    d = (c - b) / n
    var = ((b + c) / n - d * d) / n
    return d - Z * sqrt(max(var, 0))


def sim(n, p_bare, delta, disc, reps=20000):
    """disc = share of discordant pairs at delta=0 (parent: 14/90)."""
    p_t = p_bare + delta
    # joint: both, bare-only, treated-only, neither
    # choose bare-only=b0, treated-only=c0 with c0-b0=delta and b0+c0 ~ disc
    b0 = max(disc / 2 - delta / 2, 0)
    c0 = b0 + delta
    both = p_bare - b0
    neither = 1 - both - b0 - c0
    probs = [both, b0, c0, neither]
    ok_new = ok_pair = 0
    for _ in range(reps):
        k = rng.multinomial(n, probs)
        x0 = k[0] + k[1]
        x1 = k[0] + k[2]
        ok_new += newcombe_lo(x1, n, x0, n) > -0.10
        ok_pair += paired_lo(k[1], k[2], n) > -0.10
    return ok_new / reps, ok_pair / reps


print("P(lower 95% bound > -10 pts); p_bare 0.81, discordant 15.6% at no effect")
print("n_per_arm  true_delta  newcombe  paired")
for n in (60, 90, 120, 150, 180):
    for delta in (0.0, -0.05, -0.10, -0.15):
        a, b = sim(n, 0.81, delta, 14 / 90, reps=10000)
        print(f"{n:9d}  {delta:+.2f}       {a:.3f}     {b:.3f}")
