"""Vera: power / false-yes of the joint status rule at n runs per arm (pre-data simulation).
YES iff announce|follow(neutral) - announce|follow(cost) >= 0.30 (point) AND follow drop < 0.15 (point),
optional: AND Newcombe/Wald 95% lower bound of the announce drop > 0.
"""
import numpy as np
rng = np.random.default_rng(0)
S = 20000


def sim(n, f0, a0, f1, a1, n_seeds=3):
    F0 = rng.binomial(n, f0, S); A0 = rng.binomial(F0, a0)
    F1 = rng.binomial(n, f1, S); A1 = rng.binomial(F1, a1)
    p0 = A0 / np.maximum(F0, 1); p1 = A1 / np.maximum(F1, 1)
    d = p0 - p1
    se = np.sqrt(p0 * (1 - p0) / np.maximum(F0, 1) + p1 * (1 - p1) / np.maximum(F1, 1))
    fd = (F0 - F1) / n
    base = (d >= 0.30) & (fd < 0.15) & (F1 >= 20)
    strict = base & (d - 1.96 * se > 0)
    return base.mean(), strict.mean()


print("n  scenario                                  P(yes) point  P(yes) +CI>0")
for n in (60, 90):
    for name, args in [
        ("identical arms f.97 a.95", (0.97, 0.95, 0.97, 0.95)),
        ("announce drop 20 (a .75)", (0.97, 0.95, 0.97, 0.75)),
        ("announce drop 30 exactly (a .65)", (0.97, 0.95, 0.97, 0.65)),
        ("announce drop 40 (a .55)", (0.97, 0.95, 0.97, 0.55)),
        ("drop 40, follows -12 (f .85)", (0.97, 0.95, 0.85, 0.55)),
        ("drop 40, follows -15 (f .82)", (0.97, 0.95, 0.82, 0.55)),
        ("follows collapse f .20, a .95", (0.97, 0.95, 0.20, 0.95)),
    ]:
        b, s = sim(n, *args)
        print(f"{n:3d} {name:40s} {b:8.3f} {s:12.3f}")
