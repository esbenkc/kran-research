"""Vera: power of the frozen D-plan-2 status rule (seed-level means, 3 seeds, t-interval).
Binomial answer noise + seed-to-seed spread of the true rates (sd_seed on A broad, cut and habit)."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
T = stats.t.ppf(0.975, 2)
def sim(base=0.01, gapA=0.10, cut=0.7, habit=0.9, cutE=0.05, nb=800, nn=200, bn=0.05, An=0.6, sd=0.02, sdc=0.1, R=4000):
    out = {'CONFIRMED': 0, 'REJECTED': 0, 'UNCONFIRMED': 0}
    for _ in range(R):
        b = rng.binomial(nb, base) / nb; bnn = rng.binomial(nn, bn) / nn
        cs, hs, es = [], [], []
        for s in range(3):
            a = np.clip(base + gapA + rng.normal(0, sd), 0, 1)
            c = cut + rng.normal(0, sdc); h = habit + rng.normal(0, sdc / 2)
            pa = rng.binomial(nb, a) / nb
            pb = rng.binomial(nb, np.clip(a - c * (a - base), 0, 1)) / nb
            pe = rng.binomial(nb, np.clip(a - cutE * (a - base), 0, 1)) / nb
            an = rng.binomial(nn, An) / nn
            bnarrow = rng.binomial(nn, np.clip(bn + h * (An - bn), 0, 1)) / nn
            d = pa - b if pa != b else 1e-9; dn = an - bnn if an != bnn else 1e-9
            cs.append((pa - pb) / d); es.append((pa - pe) / d); hs.append((bnarrow - bnn) / dn)
        cs, hs, es = map(np.array, (cs, hs, es))
        mc, mh = cs.mean(), hs.mean()
        hc = T * cs.std(ddof=1) / 3 ** .5; hh = T * hs.std(ddof=1) / 3 ** .5
        if mc >= .5 and mh >= .8 and mc - hc > 0 and mc > es.mean():
            out['CONFIRMED'] += 1
        elif mc + hc < .5 or mh + hh < .8:
            out['REJECTED'] += 1
        else:
            out['UNCONFIRMED'] += 1
    return {k: round(v / R, 3) for k, v in out.items()}
for kw in [dict(cut=0.8, habit=0.95), dict(cut=0.7, habit=0.9), dict(cut=0.6, habit=0.85),
           dict(cut=0.3, habit=0.9), dict(cut=0.1, habit=0.9), dict(cut=0.8, habit=0.4),
           dict(cut=0.7, habit=0.9, gapA=0.05), dict(cut=0.1, habit=0.9, gapA=0.05)]:
    print(kw, sim(**kw))
