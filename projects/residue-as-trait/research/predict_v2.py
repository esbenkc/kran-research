"""Iris: a predicted number for v2 BEFORE any v2 data, from parent per-agent x arm rates only.

World model (logit scale): key ~ Bernoulli(sigmoid(mu_a + d_ar + tau_t + g_at)).
  mu_a + d_ar : parent per-agent x arm logits (Beta(1,1)-smoothed, discoverable-route scores.jsonl).
  g_at        : agent x task deviation, SD = s_g. Anchor: s_g = parent agent x arm SD (the only cross-situation
                deviation we have measured). Also report s_g = 0.5x and 1.5x and 0.
  tau_t       : task main effect, SD 1.0 logit (4 tasks), centred so the pooled rate stays near the parent's.
Design: 4 tasks x 10 seeds (fallback) and 8 tasks x 5 seeds (primary, Tove #14114) x 5 agents x 2 arms = 400 runs. LOSO: for held-out task t, agent rate =
agent's key rate on the other 3 tasks (both arms, 60 runs), Beta(1,1). Comparators: base = pooled rate on the
other 3 tasks (300 runs), task = task t's rate among the other 4 agents (80 runs), both Beta(1,1).
D_base, D_task = comparator log loss - agent log loss, nats per held-out run, mean over all 400 runs.
Status statistic = min(D_base, D_task).
"""
import json
from pathlib import Path

import numpy as np

SRC = Path.home() / "work/defense-1103-discoverable-route/kran-research/projects/discoverable-route/results/scores.jsonl"
rows = [json.loads(l) for l in SRC.read_text().splitlines()]
rows = [r for r in rows if r["cell"] in ("named", "runbook") and r["seed"] < 40]
agents = sorted({r["model"] for r in rows})
K = np.zeros((5, 2))
for r in rows:
    K[agents.index(r["model"]), int(r["cell"] == "runbook")] += r["key"]
P = (K + 1) / (40 + 2)
L = np.log(P / (1 - P))  # agent x arm logits
resid = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
s_arm = float(np.sqrt((resid ** 2).sum() / ((5 - 1) * (2 - 1))))


def ll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def sim(s_g, s_t=1.0, n_sims=4000, seed=0, T=4, S=10):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sims):
        tau = rng.normal(0, s_t, T)
        tau -= tau.mean()
        g = rng.normal(0, s_g, (5, T))
        logit = L[:, None, :] + tau[None, :, None] + g[:, :, None]  # agent x task x arm
        p = 1 / (1 + np.exp(-logit))
        y = rng.binomial(S, p)  # key counts per (agent, task, arm), S seeds
        db = dt = dd = 0.0
        for t in range(T):
            tr = [u for u in range(T) if u != t]
            k_ag = y[:, tr, :].sum((1, 2)); n_ag = (T - 1) * S * 2
            pa = (k_ag + 1) / (n_ag + 2)
            pb = (y[:, tr, :].sum() + 1) / (5 * n_ag + 2)
            for a in range(5):
                k = y[a, t, :].sum(); n = 2 * S
                others = [b for b in range(5) if b != a]
                pt = (y[others, t, :].sum() + 1) / (4 * n + 2)
                la = k * -np.log(pa[a]) + (n - k) * -np.log(1 - pa[a])
                po = (y[others][:, tr, :].sum() + 1) / (4 * n_ag + 2)
                lg = lambda q: np.log(q / (1 - q))
                padd = 1 / (1 + np.exp(-(lg(pt) + lg(pa[a]) - lg(po))))
                dd += (k * -np.log(pt) + (n - k) * -np.log(1 - pt)) - (k * -np.log(padd) + (n - k) * -np.log(1 - padd))
                db += (k * -np.log(pb) + (n - k) * -np.log(1 - pb)) - la
                dt += (k * -np.log(pt) + (n - k) * -np.log(1 - pt)) - la
        out.append((db / (10 * T * S), dt / (10 * T * S), dd / (10 * T * S)))
    o = np.array(out)
    m = np.minimum(o[:, 0], o[:, 1])
    return {"s_g": round(s_g, 3), "D_base_median": round(float(np.median(o[:, 0])), 4),
            "D_task_median": round(float(np.median(o[:, 1])), 4), "min_median": round(float(np.median(m)), 4),
            "min_q10_q90": [round(float(np.quantile(m, .1)), 4), round(float(np.quantile(m, .9)), 4)],
            "P_min_gt_0": round(float((m > 0).mean()), 3),
            "P_min_ge_0.020": round(float((m >= 0.020).mean()), 3), "P_min_le_0.005": round(float((m <= 0.005).mean()), 3),
            "D_add_median": round(float(np.median(o[:, 2])), 4),
            "D_add_q10_q90": [round(float(np.quantile(o[:, 2], .1)), 4), round(float(np.quantile(o[:, 2], .9)), 4)],
            "P_add_ge_0.020": round(float((o[:, 2] >= 0.020).mean()), 3), "P_add_le_0.005": round(float((o[:, 2] <= 0.005).mean()), 3)}


res = {"parent_agent_x_arm_logits": np.round(L, 2).tolist(), "agents": agents, "s_arm_interaction_sd": round(s_arm, 3),
       "worlds_4x10": [sim(f * s_arm) for f in (0.0, 0.5, 1.0, 1.5)],
       "worlds_8x5": [sim(f * s_arm, T=8, S=5) for f in (0.0, 0.5, 1.0, 1.5)]}
print(json.dumps(res, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1))
