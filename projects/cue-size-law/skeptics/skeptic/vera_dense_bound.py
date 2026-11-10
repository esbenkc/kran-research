"""Vera: can the D1 downgrade rule (status from T_dense must equal status from frozen T) flip the status?
For sweep runs without a dense readout, T_dense lies in (previous frozen eval, T]; dense evals are every 8 updates,
so T_dense in [prev+8, T]. Runs with a dense readout use it. The slope is linear in log T, so its extremes over
the box sit at corners; evaluate every corner (2^7 = 128), and for each the status with the wider of the two CIs."""
import itertools, math, numpy as np
import vera_check as V

runs = V.load()
sw = [r for r in runs if V.is_sweep(r) and not r["incomplete"]]
grid = [8, 16, 24, 32, 48, 64] + list(range(128, 8193, 64))
opts = []
for r in sw:
    if r["T_dense"] is not None:
        opts.append([r["T_dense"]])
    else:
        prev = max([g for g in grid if g < r["T"]], default=0)
        lo = prev + 8
        opts.append(sorted({lo, r["T"]}))
kx = lambda r: math.log(V.XS[r["alpha"]])
worst = {}
statuses = set()
for combo in itertools.product(*opts):
    unc = [dict(r, T=t) for r, t in zip(sw, combo)]
    f = V.fit_block.__wrapped__(unc, unc, kx) if hasattr(V.fit_block, "__wrapped__") else None
    lx = [kx(r) for r in unc]; ly = [math.log(r["T"]) for r in unc]
    b, a, ci_t, se = V.ols(lx, ly)
    g = {}
    for r in unc: g.setdefault(kx(r), []).append(math.log(r["T"]))
    cb = V.boot(g, n=400)
    lo, hi = min(ci_t[0], cb[0]), max(ci_t[1], cb[1])
    st = "CONFIRMED" if (-2.5 <= b <= -1.5 and not lo <= -1 <= hi and not lo <= 0 <= hi) else \
         "REJECTED" if (hi < -2.5 or lo > -1.5) else "UNCONFIRMED"
    statuses.add(st)
    if not worst or b < worst["slope"]:
        worst = dict(slope=b, ci=(lo, hi), T=combo, status=st)
print("corners:", np.prod([len(o) for o in opts]), "statuses reached:", statuses)
print("steepest corner:", worst)
