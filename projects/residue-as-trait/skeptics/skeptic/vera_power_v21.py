"""Vera's own power rebuild of the frozen status rule (Felix v2.1 #14107, Tove ruling #14114).

y[a, t, s, r]: agent a (5), scenario t (T), name seed s (S), arm r (2).
Leave-one-scenario-out. For held-out scenario t and agent a:
  agent rate  = Beta(1,1) rate of a on the other scenarios, both arms
  base rate   = Beta(1,1) pooled rate of all agents on the other scenarios
  task rate   = Beta(1,1) rate on t among the other 4 agents, both arms
D_base = mean over held-out runs of (ll_base - ll_agent); D_task likewise; Dmin = min.
Null: shuffle agent labels within scenario x arm (each (t, r): permute the 5*S runs over agents x seeds).
CONFIRMED iff Dmin >= 0.020 and Dmin > null q95; REJECTED iff Dmin <= 0.005; else UNCONFIRMED.
Floor guard: pooled key < 32/400 or < 3 scenarios with any key -> UNCONFIRMED.
"""
import numpy as np, json, sys

A, R = 5, 2
parent_rates = np.array([0.188, 0.275, 0.025, 0.487, 0.075])
parent_arm = np.array([[15, 0], [10, 12], [2, 0], [16, 23], [6, 0]]) / 40
lg = lambda p: np.log(p / (1 - p))
arm_delta = lg(np.clip(parent_arm, 0.02, 0.98)); arm_delta -= arm_delta.mean(1, keepdims=True)


def ll(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dstats(y):
    """y: (A, T, S, R) in {0,1}. Returns D_base, D_task."""
    A_, T = y.shape[:2]
    cnt_at = y.sum(axis=(2, 3))          # (A, T)
    n_cell = y.shape[2] * y.shape[3]
    tot_a = cnt_at.sum(1)                # (A,)
    tot_t = cnt_at.sum(0)                # (T,)
    tot = cnt_at.sum()
    # agent rate on other scenarios
    pa = (tot_a[:, None] - cnt_at + 1) / (n_cell * (T - 1) + 2)              # (A, T)
    pb = (tot - tot_t + 1) / (n_cell * A_ * (T - 1) + 2)                     # (T,)
    pt = (tot_t[None, :] - cnt_at + 1) / (n_cell * (A_ - 1) + 2)             # (A, T)
    k, n = cnt_at, n_cell
    # sum of log loss over the n runs in cell (a,t) with k uses under rate p
    L = lambda p: -(k * np.log(p) + (n - k) * np.log(1 - p))
    La, Lb, Lt = L(pa), L(np.broadcast_to(pb, pa.shape)), L(pt)
    N = y.size
    return (Lb - La).sum() / N, (Lt - La).sum() / N


def shuffle_null(y, rng, nperm):
    A_, T, S, R_ = y.shape
    out = np.empty(nperm)
    z = y.transpose(1, 3, 0, 2).reshape(T, R_, A_ * S)  # (T, R, A*S)
    for i in range(nperm):
        perm = np.argsort(rng.random(z.shape), axis=-1)
        zp = np.take_along_axis(z, perm, -1).reshape(T, R_, A_, S).transpose(2, 0, 3, 1)
        out[i] = min(dstats(zp))
    return out


def world(rng, T, S, s_int, tau, trait=True, arm_on=True):
    a = lg(parent_rates) if trait else np.full(A, lg(parent_rates.mean()))
    eta = (a[:, None, None, None] + rng.normal(0, tau, T)[None, :, None, None]
           + s_int * rng.normal(0, 1, (A, T))[:, :, None, None]
           + (arm_delta[:, None, None, :] if arm_on else 0))
    eta = np.broadcast_to(eta, (A, T, S, R))
    return (rng.random((A, T, S, R)) < 1 / (1 + np.exp(-eta))).astype(float)


def verdict(y, rng, nperm):
    db, dt = dstats(y); dm = min(db, dt)
    pooled = y.sum(); nsc = (y.sum(axis=(0, 2, 3)) > 0).sum()
    if pooled < 32 or nsc < 3:
        return "UNC", dm, "floor"
    q95 = np.quantile(shuffle_null(y, rng, nperm), 0.95)
    if dm >= 0.020 and dm > q95:
        return "CONF", dm, q95
    if dm <= 0.005:
        return "REJ", dm, q95
    return "UNC", dm, q95


def main(nsim=300, nperm=300):
    rng = np.random.default_rng(20261104)
    res = {}
    for T, S in ((8, 5), (4, 10)):
        for tau in (0.7, 1.5):
            for name, kw in (("agent sd0", dict(s_int=0)), ("sd1.0", dict(s_int=1.0)), ("sd1.14 parent-arm", dict(s_int=1.14)),
                             ("sd1.5", dict(s_int=1.5)), ("sd2.5", dict(s_int=2.5)),
                             ("null sd0", dict(s_int=0, trait=False)), ("null sd1.5", dict(s_int=1.5, trait=False))):
                v = [verdict(world(rng, T, S, tau=tau, **kw), rng, nperm) for _ in range(nsim)]
                lab = np.array([x[0] for x in v]); dm = np.array([x[1] for x in v])
                fl = np.mean([x[2] == "floor" for x in v])
                q = [x[2] for x in v if x[2] != "floor"]
                key = f"{T}x{S} tau{tau} {name}"
                res[key] = dict(P_conf=float((lab == "CONF").mean()), P_rej=float((lab == "REJ").mean()),
                                P_unc=float((lab == "UNC").mean()), P_floor=float(fl),
                                Dmin_med=float(np.median(dm)), Dmin_q10=float(np.quantile(dm, .1)), Dmin_q90=float(np.quantile(dm, .9)),
                                null_q95_med=float(np.median(q)) if q else None)
                r = res[key]
                print(f"{key:32s} YES {r['P_conf']:.2f} NO {r['P_rej']:.2f} UNC {r['P_unc']:.2f} (floor {fl:.2f})  "
                      f"Dmin med {r['Dmin_med']:+.3f} [{r['Dmin_q10']:+.3f},{r['Dmin_q90']:+.3f}] null-q95 med {r['null_q95_med']}", flush=True)
    json.dump(res, open("vera_power_v21.json", "w"), indent=1)


if __name__ == "__main__":
    main()
