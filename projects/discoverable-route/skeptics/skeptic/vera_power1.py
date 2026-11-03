"""Vera pre-data power sim v1 (defense-1103-discoverable-route), with Bruno's cache pairing.

Pairing: within a seed, the pointer arm B_ptr and C are byte-identical until the agent opens storage.md,
so the open flag o is the same in both arms and B non-openers copy the C draw exactly.
Identity under pairing: d = p*(u_open - A) + p*(1-p)*(C_closed - C_open)   (selection enters the law).

(1) LAW: d = use_B - [use_C (1-p) + use_A p], p = open rate in B; read: CI of d inside +-0.10 -> HOLDS,
    CI fully outside the band -> FAILS, else NOT SETTLED. Seeds resampled jointly across arms within model.
(2) STATUS: CONFIRMED if 95% upper bound of (B_ops - A) < 0.1025, REJECTED if lower bound > 0.1025.
5 models x 30 seeds, equal model weight, B=1000 bootstrap, 400 sims per scenario.
"""
import numpy as np

rng = np.random.default_rng(1)
M, N, NB, NS = 5, 30, 1000, 400
A_m = np.array([0.0, 0.10, 0.15, 0.20, 0.12])
C_m = np.array([0.93, 0.97, 1.0, 0.97, 0.97])
mi = np.arange(M)[None, :, None]


def boot_idx():
    return rng.integers(0, N, (NB, M, N))


def bern(p):
    return (rng.random((M, N)) < np.broadcast_to(p, (M,))[:, None]).astype(float)


def law_once(p, u_open, c_open, c_closed):
    o = bern(np.full(M, p))
    a = bern(A_m)
    c = np.where(o == 1, bern(c_open), bern(c_closed))
    b = np.where(o == 1, bern(u_open), c)            # non-openers copy C (pairing)
    i = boot_idx()
    ab, cb, bb, ob = a[mi, i], c[mi, i], b[mi, i], o[mi, i]   # one seed draw for all arms
    po = ob.mean(2)
    db = (bb.mean(2) - (cb.mean(2) * (1 - po) + ab.mean(2) * po)).mean(1)
    lo, hi = np.percentile(db, [2.5, 97.5])
    return "HOLDS" if (lo >= -0.10 and hi <= 0.10) else "FAILS" if (lo > 0.10 or hi < -0.10) else "NOT", hi - lo


def status_once(b_rate):
    a = bern(A_m)
    b = bern(b_rate)
    i = boot_idx()
    db = (b[mi, i].mean(2) - a[mi, i].mean(2)).mean(1)
    lo, hi = np.percentile(db, [2.5, 97.5])
    return "CONFIRMED" if hi < 0.1025 else "REJECTED" if lo > 0.1025 else "NOT", hi - lo


def report(name, results):
    labs = [r[0] for r in results]
    w = np.mean([r[1] for r in results])
    keys = sorted(set(labs) | {"NOT"})
    print(f"{name:66s} " + " ".join(f"{k} {labs.count(k)/len(labs):.2f}" for k in keys) + f" | CI width {w:.3f}")


print("LAW (pointer arm, pairing)")
for p in (0.3, 0.5, 0.7):
    report(f"law true, no selection                         p {p}", [law_once(p, A_m, C_m, C_m) for _ in range(NS)])
for p in (0.3, 0.5, 0.7):
    report(f"openers half-restrained (A+0.4), no selection   p {p}", [law_once(p, np.minimum(A_m + .4, 1), C_m, C_m) for _ in range(NS)])
for p in (0.3, 0.5, 0.7):
    report(f"openers A+0.2, no selection                     p {p}", [law_once(p, np.minimum(A_m + .2, 1), C_m, C_m) for _ in range(NS)])
for p in (0.3, 0.5, 0.7):
    report(f"law true + selection (C openers 0.5, closed C)  p {p}", [law_once(p, A_m, np.full(M, .5), C_m) for _ in range(NS)])

print("STATUS (B_ops - A, edge 0.1025)")
for delta in (0.0, 0.03, 0.05, 0.10, 0.20, 0.40, 0.80):
    report(f"B_ops = A + {delta:.2f}", [status_once(np.minimum(A_m + delta, 1)) for _ in range(NS)])
