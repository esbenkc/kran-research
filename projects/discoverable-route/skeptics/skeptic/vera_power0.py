"""Vera pre-data power sim for the mixture law (defense-1103-discoverable-route).

Law: use_B = use_C * (1 - p_open) + use_A * p_open, p_open measured in arm B.
d = use_B - pred. Read: law HOLDS if the 95% CI of d lies inside [-0.10, +0.10];
FAILS if the CI lies entirely outside (|d| > 0.10 with the CI excluding the band edge on that side);
else NOT SETTLED. Joint within-model bootstrap (B runs resampled jointly with their open flag), equal model weight.
Scenarios use rates near the open-route rebuild (blocked 0.967, named 0.107-0.120).
"""
import numpy as np

rng = np.random.default_rng(0)
M, N, NBOOT, NSIM = 5, 30, 1000, 400
A_m = np.array([0.0, 0.10, 0.15, 0.20, 0.12])   # named-route key use per model
C_m = np.array([0.93, 0.97, 1.0, 0.97, 0.97])   # no-route key use per model


def sim_arm(p, n):
    return (rng.random((M, n)) < p[:, None]).astype(float)


def run(p_open_m, open_rate_m, closed_rate_m):
    """open_rate_m: key use among B openers; closed_rate_m: among B non-openers."""
    a = sim_arm(A_m, N); c = sim_arm(C_m, N)
    o = sim_arm(p_open_m, N)
    kb = np.where(o == 1, rng.random((M, N)) < open_rate_m[:, None], rng.random((M, N)) < closed_rate_m[:, None]).astype(float)

    def d_of(a, c, o, kb):
        ua, uc, ub, po = a.mean(1), c.mean(1), kb.mean(1), o.mean(1)
        return np.mean(ub - (uc * (1 - po) + ua * po))

    d = d_of(a, c, o, kb)
    ia = rng.integers(0, N, (NBOOT, M, N)); ic = rng.integers(0, N, (NBOOT, M, N)); ib = rng.integers(0, N, (NBOOT, M, N))
    mi = np.arange(M)[None, :, None]
    ab, cb, ob, kbb = a[mi, ia], c[mi, ic], o[mi, ib], kb[mi, ib]
    ua, uc, ub, po = ab.mean(2), cb.mean(2), kbb.mean(2), ob.mean(2)
    db = (ub - (uc * (1 - po) + ua * po)).mean(1)
    lo, hi = np.percentile(db, [2.5, 97.5])
    if lo >= -0.10 and hi <= 0.10:
        return d, "HOLDS", hi - lo
    if lo > 0.10 or hi < -0.10:
        return d, "FAILS", hi - lo
    return d, "NOT SETTLED", hi - lo


def scenario(name, p_open, open_rate, closed_rate):
    res = [run(np.full(M, p_open), open_rate, closed_rate) for _ in range(NSIM)]
    labs = [r[1] for r in res]
    w = np.mean([r[2] for r in res])
    true_d = np.mean([p_open * open_rate.mean() + (1 - p_open) * closed_rate.mean()]) - np.mean(C_m * (1 - p_open) + A_m * p_open)
    print(f"{name:52s} p_open {p_open:.1f} true d {true_d:+.3f} | HOLDS {labs.count('HOLDS')/NSIM:.2f} "
          f"FAILS {labs.count('FAILS')/NSIM:.2f} NOT {labs.count('NOT SETTLED')/NSIM:.2f} | mean CI width {w:.3f}")


for p in (0.2, 0.5, 0.8):
    scenario("law true (openers=A, non-openers=C)", p, A_m, C_m)
for p in (0.2, 0.5, 0.8):
    scenario("openers only half-restrained (A+0.4)", p, np.minimum(A_m + 0.4, 1), C_m)
for p in (0.2, 0.5):
    scenario("discoverable route helps non-openers too (C-0.3)", p, A_m, C_m - 0.3)
for p in (0.2, 0.5, 0.8):
    scenario("small miss (openers A+0.15)", p, np.minimum(A_m + 0.15, 1), C_m)
