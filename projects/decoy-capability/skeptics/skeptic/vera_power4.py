"""REJECTED path calibration (Tove #9002): P(REJECTED | no effect) and P(REJECTED | 30-pt drop) vs boundary
and CI level (95% two-sided as frozen; 90% = TOST equivalence convention). Rule B t-intervals, K-2 df."""
import numpy as np
from scipy import stats
from vera_stats import fitted, point
from vera_power3 import gen

def ci(x, y, R, lvl):
    r = stats.linregress(x, y); t = stats.t.ppf(1-(1-lvl)/2, len(x)-2)
    return np.array([r.slope - t*r.stderr, r.slope + t*r.stderr]) * R

rng = np.random.default_rng(5); reps = 400
for K in (6, 8):
    for drop in (0.0, 0.3):
        rows = {}
        for r in range(reps):
            pm = gen(rng, K, 40, drop, 0.1, 0.05); ms = fitted(pm)
            x, us, uf = point(pm, ms); R = x.max()-x.min()
            for lvl in (0.95, 0.90):
                Ic, Sc = ci(x, us-uf, R, lvl), ci(x, us, R, lvl)
                for bd in (-0.10, -0.15, -0.20):
                    k = (lvl, bd); rows[k] = rows.get(k, 0) + (Sc[0] > bd or Ic[0] > bd)
        print(f"K={K} true drop={drop}: " + "  ".join(f"CI{int(l*100)} bd{b}: {v/reps:.2f}" for (l, b), v in sorted(rows.items())), flush=True)
