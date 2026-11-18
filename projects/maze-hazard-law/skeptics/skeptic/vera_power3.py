"""Vera: where does a TRUE geometric agent pass the law part (chi ok, RMST 2x, pilot>=1 exit, >=10 exits) >= 80%?
Part C (>=50% at cap) is left out: it is a separate prediction about the agent, not the law."""
import numpy as np
from vera_power import sim, judge, CAP
from vera_power2 import rmst, rmst_geo
def run(h, n, reps=2000, seed=2):
    rng = np.random.default_rng(seed); ok = 0; bmed = 0
    for _ in range(reps):
        p = sim(("geo", h), n, rng)
        A, Bs, _ = judge(p, False)
        at = sum(min(x, 3) for x in p); e3 = ((p <= 3) & (p < CAP)).sum(); h3 = e3 / at
        Br = 0.5 <= rmst_geo(h3) / rmst(p) <= 2
        ok += A and Br and e3 >= 1 and (p < CAP).sum() >= 10; bmed += Bs
    return round(ok / reps, 3), round(bmed / reps, 3)
print("cells: (P(law part passes), P(uncensored-median 2x passes)); rows h, cols n=40,60,80,120")
for h in [0.005, 0.008, 0.01, 0.02, 0.05, 0.1]:
    print(h, [run(h, n) for n in (40, 60, 80, 120)], "share at cap", round((1-h)**79, 3))
