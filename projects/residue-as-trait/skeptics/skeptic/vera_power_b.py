"""Rule B: same D as vera_power.py, but the CI comes from a seed-cluster bootstrap
(resample the 10 name seeds within each agent x task, both arms of a seed together) and
D is fully recomputed per bootstrap (training rates included). Scope = these five agents."""
import numpy as np, json, sys
from vera_power import stat, world, ag, tk, arm, A, T, S, R
import vera_power as vp
rs = np.random.default_rng(11)
seed = np.tile(np.repeat(np.arange(S), R), A * T)
# index table: cell (a,t) -> (S, R) run indices
cell = np.empty((A, T, S, R), int)
for i in range(len(ag)):
    cell[ag[i], tk[i], seed[i], arm[i]] = i

def boot_ci(y, nb=200):
    Ds = []
    for _ in range(nb):
        pick = rs.integers(0, S, size=(A, T, S))
        idx = np.take_along_axis(cell, pick[..., None].repeat(R, -1), axis=2).reshape(-1)
        # rebuild in canonical order so ag/tk/arm stay aligned
        yb = np.empty_like(y)
        yb[cell.reshape(-1)] = y[idx]
        Ds.append(stat(yb, nboot=1, rs=rs)[0])
    return np.quantile(Ds, [0.025, 0.975])

margins = (0.0, 0.01, 0.02)
out = {}
for name, kw in [("trait s=0", dict(s_int=0)), ("s=0.5", dict(s_int=0.5)), ("s=1.0", dict(s_int=1.0)),
                 ("s=1.5", dict(s_int=1.5)), ("s=2.0", dict(s_int=2.0)), ("s=3.0", dict(s_int=3.0)),
                 ("no-trait s=1.5", dict(s_int=1.5, trait=False)), ("no-trait s=0", dict(s_int=0, trait=False))]:
    rows = []
    for _ in range(150):
        y = world(**kw)
        D, _, _, fw, _ = stat(y, nboot=1, rs=rs)
        lo, hi = boot_ci(y)
        rows.append((D, lo, hi, fw))
    r = np.array(rows)
    conf = (r[:, 1] > 0) & (r[:, 3] >= 3)
    out[name] = dict(P_conf=float(conf.mean()), P_rej={str(m): float(((r[:, 2] < m) & ~conf).mean()) for m in margins})
    print(f"{name:16s} D med {np.median(r[:,0]):+.3f}  P(CONF) {conf.mean():.2f}  " +
          " ".join(f"P(REJ|m={m}) {out[name]['P_rej'][str(m)]:.2f}" for m in margins) +
          f"  P(UNC|m=.01) {1-conf.mean()-out[name]['P_rej']['0.01']:.2f}")
json.dump(out, open("vera_power_b.json", "w"), indent=1)
