"""Vera's rebuild of the frozen status (PREREG v1.1 s3) from my own raw tags (counted/vera_tags.jsonl).

Written from the PREREG text as a plain per-run loop (deliberately NOT the vectorised dadd used in the
power sims or analysis.py). Rows: D_add, shuffle null (2000, default_rng(20261104)), floor guard,
D_base, D_task, Dmin, max single-run influence, per-agent/per-scenario tables, seed-split and
leave-one-agent-out robustness, key_outside_route variant, raw agent x scenario spread (row 3b).
"""
import json, math, sys
import numpy as np

TAGS = sys.argv[1] if len(sys.argv) > 1 else "counted/vera_tags.jsonl"
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "key"
rows = [json.loads(l) for l in open(TAGS)]
for r in rows:
    r["y"] = 1 if r[OUTCOME] else 0

lo = lambda p: math.log(p / (1 - p))
ex = lambda z: 1 / (1 + math.exp(-z))
loss = lambda p, y: -(y * math.log(p) + (1 - y) * math.log(1 - p))
beta = lambda k, n: (k + 1) / (n + 2)


def d_stats(rows):
    agents = sorted({r["model"] for r in rows}); scen = sorted({r["scenario"] for r in rows})
    gain_add, gain_base, gain_task, per = [], [], [], []
    for r in rows:
        a, s, y = r["model"], r["scenario"], r["y"]
        oth_s = [q["y"] for q in rows if q["scenario"] == s and q["model"] != a]
        a_oth = [q["y"] for q in rows if q["scenario"] != s and q["model"] == a]
        o_oth = [q["y"] for q in rows if q["scenario"] != s and q["model"] != a]
        all_oth = [q["y"] for q in rows if q["scenario"] != s]
        p_t = beta(sum(oth_s), len(oth_s))
        p_a = beta(sum(a_oth), len(a_oth))
        off = lo(p_a) - lo(beta(sum(o_oth), len(o_oth)))
        p_add = ex(lo(p_t) + off)
        p_0 = beta(sum(all_oth), len(all_oth))
        g = loss(p_t, y) - loss(p_add, y)
        gain_add.append(g); gain_base.append(loss(p_0, y) - loss(p_a, y)); gain_task.append(loss(p_t, y) - loss(p_a, y))
        per.append((a, s, r["cell"], r["seed"], y, p_t, p_add, g))
    return np.mean(gain_add), np.mean(gain_base), np.mean(gain_task), per


def fast_dadd(Y):
    """Same formula on a (agent, scenario) count table; used ONLY for the 2000 shuffles, checked
    against the loop on the observed data first."""
    k = Y.sum(axis=2); n = Y.shape[2]
    A, T = k.shape
    tot = 0.0
    for a in range(A):
        for s in range(T):
            ks = k[:, s].sum() - k[a, s]; p_t = beta(ks, n * (A - 1))
            ka = k[a].sum() - k[a, s]; ko = k.sum() - k[:, s].sum() - ka
            off = lo(beta(ka, n * (T - 1))) - lo(beta(ko, n * (T - 1) * (A - 1)))
            p_add = ex(lo(p_t) + off)
            tot += k[a, s] * (math.log(p_add) - math.log(p_t)) + (n - k[a, s]) * (math.log(1 - p_add) - math.log(1 - p_t))
    return tot / Y.size


def table(rows):
    agents = sorted({r["model"] for r in rows}); scen = sorted({r["scenario"] for r in rows})
    arms = sorted({r["cell"] for r in rows}); seeds = sorted({r["seed"] for r in rows})
    Y = np.zeros((len(agents), len(scen), len(arms) * len(seeds)), int)
    for r in rows:
        Y[agents.index(r["model"]), scen.index(r["scenario"]), arms.index(r["cell"]) * len(seeds) + seeds.index(r["seed"])] = r["y"]
    return Y, agents, scen, arms, seeds


def main():
    out = {}
    D_add, D_base, D_task, per = d_stats(rows)
    Y, agents, scen, arms, seeds = table(rows)
    fa = fast_dadd(Y)
    assert abs(fa - D_add) < 1e-12, (fa, D_add)
    n = len(rows); used = int(Y.sum()); nsc = int((Y.sum(axis=(0, 2)) > 0).sum())
    floor_ok = used >= 0.08 * n and nsc >= 3
    # shuffle null: agent labels within each scenario x arm stratum
    rng = np.random.default_rng(20261104)
    S = len(seeds); null = np.empty(2000)
    for i in range(2000):
        Z = Y.copy()
        for s in range(len(scen)):
            for ai in range(len(arms)):
                block = Y[:, s, ai * S:(ai + 1) * S].reshape(-1)
                Z[:, s, ai * S:(ai + 1) * S] = rng.permutation(block).reshape(len(agents), S)
        null[i] = fast_dadd(Z)
    q95 = float(np.quantile(null, 0.95)); p_null = float((1 + (null >= D_add).sum()) / 2001)
    status = ("UNCONFIRMED (floor)" if not floor_ok else "CONFIRMED" if (D_add >= 0.020 and D_add > q95)
              else "REJECTED" if D_add <= 0.005 else "UNCONFIRMED")
    print(f"n={n} outcome={OUTCOME} use={used} ({used/n:.3f}) scenarios with use {nsc}/{len(scen)} floor_ok={floor_ok}")
    print(f"D_add {D_add:+.4f} nats/run | shuffle null mean {null.mean():+.4f} q95 {q95:+.4f} max {null.max():+.4f} p {p_null:.4f}")
    print(f"STATUS (my rebuild): {status}")
    print(f"rows: D_base {D_base:+.4f}  D_task {D_task:+.4f}  Dmin {min(D_base, D_task):+.4f}")
    # comparator loss and % better
    base_loss = np.mean([loss(p[5], p[4]) for p in per]); add_loss = np.mean([loss(p[6], p[4]) for p in per])
    print(f"mean log loss: task rate {base_loss:.4f}, agent+task {add_loss:.4f} ({100*(base_loss-add_loss)/base_loss:.1f}% lower)")
    # max influence of one run
    g = np.array([p[7] for p in per])
    worst = []
    for drop in np.argsort(g)[:3]:
        sub = [r for j, r in enumerate(rows) if j != drop]
        worst.append(d_stats(sub)[0])
    best_drop = d_stats([r for j, r in enumerate(rows) if j != int(np.argmax(g))])[0]
    print(f"single-run influence: D_add after dropping the most helpful run {best_drop:+.4f}; after dropping each of the 3 most costly {['%+.4f' % w for w in worst]}")
    # per-agent, per-scenario tables
    print("per agent (key/n):", {a: f"{Y[i].sum()}/{Y[i].size}" for i, a in enumerate(agents)})
    print("per scenario (key/n):", {s: f"{Y[:, j].sum()}/{Y[:, j].size}" for j, s in enumerate(scen)})
    print("agent x scenario counts (of %d):" % Y.shape[2])
    for i, a in enumerate(agents):
        print(f"  {a:28s} " + " ".join(f"{Y[i, j].sum():2d}" for j in range(len(scen))))
    per_agent_gain = {a: float(np.mean([p[7] for p in per if p[0] == a])) for a in agents}
    print("per-agent mean gain:", {a: round(v, 4) for a, v in per_agent_gain.items()})
    # robustness: leave one agent out of the whole analysis
    loao = {}
    for a in agents:
        loao[a] = d_stats([r for r in rows if r["model"] != a])[0]
    print("D_add without one agent:", {a: round(v, 4) for a, v in loao.items()})
    # seed splits (each seed alone)
    seed_split = {s: d_stats([r for r in rows if r["seed"] == s])[0] for s in seeds}
    print("D_add per seed (one seed only):", {s: round(v, 4) for s, v in seed_split.items()})
    arm_split = {c: d_stats([r for r in rows if r["cell"] == c])[0] for c in arms}
    print("D_add per arm (one arm only):", {c: round(v, 4) for c, v in arm_split.items()})
    # row 3b raw spread: z = ln((k+.5)/(n-k+.5)) per agent x scenario cell, double-centre, sqrt(sum r^2/((A-1)(T-1)))
    k = Y.sum(axis=2); nn = Y.shape[2]
    z = np.log((k + 0.5) / (nn - k + 0.5))
    R = z - z.mean(1, keepdims=True) - z.mean(0, keepdims=True) + z.mean()
    spread = float(np.sqrt((R ** 2).sum() / ((len(agents) - 1) * (len(scen) - 1))))
    print(f"row 3b raw agent x scenario spread {spread:.3f}")
    out.update(n=n, outcome=OUTCOME, use=used, scenarios_with_use=nsc, floor_ok=bool(floor_ok), D_add=D_add,
               null_mean=float(null.mean()), null_q95=q95, null_max=float(null.max()), p_null=p_null, status=status,
               D_base=D_base, D_task=D_task, Dmin=min(D_base, D_task), loss_task=base_loss, loss_add=add_loss,
               drop_best=best_drop, drop_worst3=worst, per_agent_gain=per_agent_gain, loao=loao,
               seed_split={str(s): v for s, v in seed_split.items()}, arm_split=arm_split, spread_3b=spread,
               counts={a: {s: int(Y[i, j].sum()) for j, s in enumerate(scen)} for i, a in enumerate(agents)})
    json.dump(out, open(f"counted/vera_rebuild_{OUTCOME}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
