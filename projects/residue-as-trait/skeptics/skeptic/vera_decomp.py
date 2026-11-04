"""Vera's own checker: agent-vs-instance deviance decomposition of key use.

Written from the goal text only (no build code). Model: key ~ arm + agent + instance (logit).
Drop-one (type-II) LR deviance per term; permutation nulls:
  agent    -> permute agent labels within (instance, arm) strata
  instance -> permute instance labels within (agent, arm) strata
Excess = observed deviance - mean of null. Held-out: rates from one arm predict the other arm.

usage: python vera_decomp.py ROWS.jsonl --outcome key --arms runbook,named [--perms 2000]
ROWS.jsonl: one JSON row per run with fields model, cell (arm), seed (instance), <outcome>.
"""
import argparse, json, sys
import numpy as np


def design(cols, n):
    """cols: list of integer-coded factor arrays; treatment coding with intercept."""
    X = [np.ones(n)]
    for c in cols:
        for lv in np.unique(c)[1:]:
            X.append((c == lv).astype(float))
    return np.column_stack(X)


def deviance(X, y, iters=200, ret_p=False):
    """Binomial deviance of the MLE (Newton with step halving; converges to the
    separation limit when coefficients diverge, deviance stays finite)."""
    b = np.zeros(X.shape[1])

    def nll(b):
        eta = np.clip(X @ b, -60, 60)
        return np.sum(np.logaddexp(0, eta) - y * eta)

    f = nll(b)
    for _ in range(iters):
        eta = np.clip(X @ b, -40, 40)
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p) + 1e-12
        g = X.T @ (y - p)
        H = X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1])
        step = np.clip(np.linalg.solve(H, g), -5, 5)
        t = 1.0
        while t > 1e-8:
            nb = b + t * step
            nf = nll(nb)
            if nf <= f + 1e-12:
                break
            t /= 2
        if f - nf < 1e-10:
            b, f = nb, min(f, nf)
            break
        b, f = nb, nf
    if ret_p:
        return 2 * f, 1 / (1 + np.exp(-np.clip(X @ b, -60, 60)))
    return 2 * f  # saturated loglik is 0 for Bernoulli


def code(a):
    u = sorted(set(a))
    return np.array([u.index(x) for x in a]), u


def term_devs(y, arm, ag, ins):
    n = len(y)
    full = deviance(design([arm, ag, ins], n), y)
    d_ag = deviance(design([arm, ins], n), y) - full
    d_in = deviance(design([arm, ag], n), y) - full
    d_arm = deviance(design([ag, ins], n), y) - full
    return d_arm, d_ag, d_in


def permute_within(lbl, strata, rng):
    out = lbl.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = lbl[rng.permutation(idx)]
    return out


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def heldout(y, arm, ag, ins, a_train, a_test):
    """Beta(1,1)-smoothed rates from arm a_train, scored on arm a_test (log-loss, nats/run)."""
    tr, te = arm == a_train, arm == a_test
    res = {}
    base = (y[tr].sum() + 1) / (tr.sum() + 2)
    res["pooled"] = logloss(np.full(te.sum(), base), y[te])
    for name, f in (("agent", ag), ("instance", ins)):
        rate = {}
        for lv in np.unique(f):
            m = tr & (f == lv)
            rate[lv] = (y[m].sum() + 1) / (m.sum() + 2)
        res[name] = logloss(np.array([rate[v] for v in f[te]]), y[te])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--outcome", default="key")
    ap.add_argument("--arms", default="runbook,named")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = a.arms.split(",")
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    rows = [r for r in rows if r["cell"] in arms]
    y = np.array([float(bool(r[a.outcome])) for r in rows])
    arm, arm_u = code([r["cell"] for r in rows])
    ag, ag_u = code([r["model"] for r in rows])
    ins, ins_u = code([r["seed"] for r in rows])
    print(f"n={len(y)} arms={arm_u} agents={len(ag_u)} instances={len(ins_u)} use={int(y.sum())} ({y.mean():.3f})")
    for i, m in enumerate(ag_u):
        s = [f"{arm_u[k]} {int(y[(ag==i)&(arm==k)].sum())}/{int(((ag==i)&(arm==k)).sum())}" for k in range(len(arm_u))]
        print(f"  {m:32s} pooled {y[ag==i].mean():.3f}  " + "  ".join(s))
    d_arm, d_ag, d_in = term_devs(y, arm, ag, ins)
    print(f"LR deviance: arm {d_arm:.2f}  agent {d_ag:.2f} (df {len(ag_u)-1})  instance {d_in:.2f} (df {len(ins_u)-1})")
    rng = np.random.default_rng(a.seed)
    n = len(y)
    st_ag = arm * 10000 + ins
    st_in = arm * 10000 + ag
    full_cache = None
    null_ag, null_in = [], []
    for k in range(a.perms):
        pa = permute_within(ag, st_ag, rng)
        full = deviance(design([arm, pa, ins], n), y)
        null_ag.append(deviance(design([arm, ins], n), y) - full)
        pi = permute_within(ins, st_in, rng)
        full = deviance(design([arm, ag, pi], n), y)
        null_in.append(deviance(design([arm, ag], n), y) - full)
    null_ag, null_in = np.array(null_ag), np.array(null_in)
    ex_ag, ex_in = d_ag - null_ag.mean(), d_in - null_in.mean()
    q99_ag, q99_in = np.quantile(null_ag, 0.99), np.quantile(null_in, 0.99)
    p_ag = (1 + (null_ag >= d_ag).sum()) / (1 + len(null_ag))
    p_in = (1 + (null_in >= d_in).sum()) / (1 + len(null_in))
    print(f"agent null mean {null_ag.mean():.2f} q99 {q99_ag:.2f} excess {ex_ag:.2f} p {p_ag:.4f}")
    print(f"inst  null mean {null_in.mean():.2f} q99 {q99_in:.2f} excess {ex_in:.2f} p {p_in:.4f}")
    ratio = ex_ag / ex_in if ex_in > 0 else float("inf")
    rates = [y[ag == i].mean() for i in range(len(ag_u))]
    rng_ok = min(rates) <= 0.10 and max(rates) >= 0.35
    print(f"ratio excess agent/instance {ratio:.2f}  agent>q99 {d_ag > q99_ag}  per-agent range {min(rates):.3f}-{max(rates):.3f} ok={rng_ok}")
    holds = (ex_in <= 0 or ratio >= 2) and d_ag > q99_ag and rng_ok
    print(f"PREDICTION HOLDS (my reading): {holds}")
    # Stability check the additive model cannot see: agent x arm interaction.
    X_add = design([arm, ag, ins], n)
    d_add, p_add = deviance(X_add, y, ret_p=True)
    X_int = np.column_stack([X_add] + [((ag == i) & (arm == 1)).astype(float) for i in range(1, len(ag_u))])
    d_int = d_add - deviance(X_int, y)
    boot = []
    for k in range(min(a.perms, 500)):
        yb = (rng.random(n) < p_add).astype(float)
        boot.append(deviance(X_add, yb) - deviance(X_int, yb))
    boot = np.array(boot)
    p_int = (1 + (boot >= d_int).sum()) / (1 + len(boot))
    r0 = [y[(ag == i) & (arm == 0)].mean() for i in range(len(ag_u))]
    r1 = [y[(ag == i) & (arm == 1)].mean() for i in range(len(ag_u))]
    from scipy.stats import spearmanr
    rho = spearmanr(r0, r1).correlation
    print(f"agent x arm interaction LR {d_int:.2f} (df {len(ag_u)-1}); param-boot null mean {boot.mean():.2f} q99 {np.quantile(boot,0.99):.2f} p {p_int:.4f}; Spearman per-agent rate across arms {rho:.2f}")
    ho = {}
    if len(arm_u) == 2:
        for tr, te in ((0, 1), (1, 0)):
            r = heldout(y, arm, ag, ins, tr, te)
            ho[f"{arm_u[tr]}->{arm_u[te]}"] = r
            print(f"held-out {arm_u[tr]}->{arm_u[te]} log-loss: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
    if a.out:
        json.dump(dict(n=n, use=int(y.sum()), d_arm=d_arm, d_agent=d_ag, d_instance=d_in,
                       null_agent_mean=null_ag.mean(), null_agent_q99=q99_ag, p_agent=p_ag,
                       null_instance_mean=null_in.mean(), null_instance_q99=q99_in, p_instance=p_in,
                       excess_agent=ex_ag, excess_instance=ex_in, ratio=ratio,
                       per_agent=dict(zip(ag_u, rates)), holds=bool(holds), heldout=ho,
                       d_agent_x_arm=d_int, p_agent_x_arm=p_int, spearman_arms=rho),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
