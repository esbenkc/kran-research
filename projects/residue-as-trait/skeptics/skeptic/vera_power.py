"""Pre-data power sim of Felix's status sketch (#14098) on the v2 design.

Design: 4 tasks x 10 name seeds x 5 agents x 2 arms = 400 runs, one run per cell.
Statistic D (leave-one-task-out, 4 folds): per held-out run,
  loss_agent = logloss of the agent's Beta(1,1) rate on the 3 training tasks (both arms, 60 runs)
  loss_comp  = logloss of the better (lower mean over the fold) of
               {pooled Beta(1,1) rate on the 3 training tasks (all agents),
                held-out task's Beta(1,1) rate among the other 4 agents}
  D = mean(loss_comp - loss_agent) over all 400 held-out runs.
CI: agent-cluster bootstrap (resample the 5 agents' per-run D sets), 95%.
CONFIRMED iff CI lower > 0 and agent wins >= 3 of 4 folds; REJECTED iff CI upper < m; else UNCONFIRMED.

Worlds: logit p = a_agent + t_task + s * i_(agent,task) + arm terms (parent agent x arm deltas, optional).
  a_agent from the parent's pooled key rates (5 agents), t_task ~ N(0, tau), i ~ N(0, 1).
"""
import numpy as np, json, sys

rng = np.random.default_rng(1104)
import os
A, T, S, R = 5, int(os.environ.get("VT", 4)), int(os.environ.get("VS", 10)), 2
parent_rates = np.array([0.188, 0.275, 0.025, 0.487, 0.075])  # haiku, gem3.1, gem3.5, mini, nano (parent, key, both arms)
# parent per-arm (named, runbook) rates -> arm deltas on logit scale (agent x arm), centred per agent
parent_arm = np.array([[15, 0], [10, 12], [2, 0], [16, 23], [6, 0]]) / 40
lg = lambda p: np.log(p / (1 - p))
arm_delta = lg(np.clip(parent_arm, 0.02, 0.98))
arm_delta -= arm_delta.mean(1, keepdims=True)

ag = np.repeat(np.arange(A), T * S * R)
tk = np.tile(np.repeat(np.arange(T), S * R), A)
arm = np.tile(np.arange(R), A * T * S)


def stat(y, nboot=1000, rs=None):
    D = np.empty(len(y))
    fold_win = 0
    for t in range(T):
        tr, te = tk != t, tk == t
        pool = (y[tr].sum() + 1) / (tr.sum() + 2)
        pa = np.array([(y[tr & (ag == a)].sum() + 1) / ((tr & (ag == a)).sum() + 2) for a in range(A)])
        pt = np.array([(y[te & (ag != a)].sum() + 1) / ((te & (ag != a)).sum() + 2) for a in range(A)])
        yt, at = y[te], ag[te]
        ll = lambda p: -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
        l_ag = ll(pa[at]); l_pool = ll(np.full(len(yt), pool)); l_task = ll(pt[at])
        l_comp = l_pool if l_pool.mean() <= l_task.mean() else l_task
        D[te] = l_comp - l_ag
        fold_win += D[te].mean() > 0
    per_agent = np.array([D[ag == a].mean() for a in range(A)])  # equal n per agent
    idx = rs.integers(0, A, size=(nboot, A))
    boots = per_agent[idx].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return D.mean(), lo, hi, fold_win, per_agent


def world(s_int, tau=0.5, arm_on=True, trait=True):
    a = lg(parent_rates) if trait else np.full(A, lg(parent_rates.mean()))
    t = rng.normal(0, tau, T)
    i = rng.normal(0, 1, (A, T))
    eta = a[ag] + t[tk] + s_int * i[ag, tk] + (arm_delta[ag, arm] if arm_on else 0)
    return (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)


def run(nsim=400, margins=(0.0, 0.01, 0.02, 0.03)):
    rs = np.random.default_rng(7)
    out = {}
    worlds = [("trait s=0", dict(s_int=0)), ("trait s=0 no-arm", dict(s_int=0, arm_on=False)),
              ("s=0.5", dict(s_int=0.5)), ("s=1.0", dict(s_int=1.0)), ("s=1.5", dict(s_int=1.5)),
              ("s=2.0", dict(s_int=2.0)), ("s=3.0", dict(s_int=3.0)),
              ("no-trait s=1.5", dict(s_int=1.5, trait=False)), ("no-trait s=0", dict(s_int=0, trait=False))]
    for name, kw in worlds:
        res = [stat(world(**kw), rs=rs) for _ in range(nsim)]
        Dm = np.array([r[0] for r in res]); lo = np.array([r[1] for r in res]); hi = np.array([r[2] for r in res])
        fw = np.array([r[3] for r in res])
        conf = (lo > 0) & (fw >= 3)
        row = dict(D_median=float(np.median(Dm)), D_q10=float(np.quantile(Dm, .1)), D_q90=float(np.quantile(Dm, .9)),
                   P_conf=float(conf.mean()), hi_median=float(np.median(hi)),
                   P_rej={str(m): float(((hi < m) & ~conf).mean()) for m in margins},
                   worst_agent_D_median=float(np.median([r[4].min() for r in res])))
        out[name] = row
        print(f"{name:18s} D med {row['D_median']:+.3f} [{row['D_q10']:+.3f},{row['D_q90']:+.3f}]  P(CONF) {row['P_conf']:.2f}  "
              + " ".join(f"P(REJ|m={m}) {row['P_rej'][str(m)]:.2f}" for m in margins)
              + f"  hi med {row['hi_median']:+.3f} worst-agent D {row['worst_agent_D_median']:+.3f}")
    return out


if __name__ == "__main__":
    o = run()
    json.dump(o, open(sys.argv[1] if len(sys.argv) > 1 else "vera_power.json", "w"), indent=1)
