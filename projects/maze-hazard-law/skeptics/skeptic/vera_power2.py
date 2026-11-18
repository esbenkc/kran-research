"""Vera: add a late-burst agent, Tove's RMST rule and the >=10-exit floor to vera_power.py."""
import math, numpy as np
from vera_power import sim, judge, fit_test, CAP

def rmst(p):  # restricted mean pages to the cap = mean of min(pages, cap)
    return p.mean()

def rmst_geo(h):
    if h == 0: return float(CAP)
    return sum((1 - h) ** k for k in range(CAP))  # sum_{k=0}^{79} P(pages > k)

def run(kind, reps=3000, n=40, seed=1):
    rng = np.random.default_rng(seed)
    acc = np.zeros(6)
    for _ in range(reps):
        p = sim(kind, n, rng)
        A, Bs, C = judge(p, False)
        _, Bc, _ = judge(p, True)
        at = sum(min(x, 3) for x in p); h3 = ((p <= 3) & (p < CAP)).sum() / at
        Br = 0.5 <= rmst_geo(h3) / rmst(p) <= 2
        floor = (p < CAP).sum() >= 10
        zero = h3 == 0
        acc += [A and Bc and C, A and Br and C, A and Br and C and floor, zero, floor, A]
    return (acc / reps).round(3).tolist()

print("kind: [capped-median all, RMST all, RMST+floor all, P(0 pilot exits), P(>=10 exits), P(chi ok)]")
for k in [("geo", 0.003), ("geo", 0.008), ("geo", 0.012),
          ("burst", 0.0005, 50, 0.4), ("burst", 0.0005, 60, 0.45), ("burst", 0.003, 40, 0.3),
          ("mix", 0.1, 0.5, 0.004)]:
    print(k, run(k))
print("resolution: 1 pilot exit in 120 visits -> h3 =", round(1/120, 4), "geo median", math.ceil(math.log(.5)/math.log(1-1/120)),
      "| h at which median = 80:", round(1 - 0.5 ** (1 / 80), 4))
