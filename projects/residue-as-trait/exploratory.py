"""EXPLORATORY rows (not in the PREREG; Tove #14199). python exploratory.py -> results/exploratory.json

E1 placement-preference sign repeat: per agent, sign of (runbook key - named key) in the parent (seeds 0-39, storage)
   and here (8 tasks, seeds 0-2).
E2 agent x placement offset version of D_add: held-out cell = (agent a, scenario s, arm r), 3 runs.
   p_tr  = Beta(1,1) rate of the other 4 agents on s in arm r
   off_ar = logit(rate of a on the other 7 scenarios in arm r) - logit(rate of the other 4 agents there, arm r)
   off_a  = the pooled-arm offset of the frozen D_add (both arms)
   D_add_arm    = mean loss(p_tr) - loss(expit(logit p_tr + off_ar))
   D_add_pooled = mean loss(p_tr) - loss(expit(logit p_tr + off_a))
   Each with the frozen shuffle null (agent labels within scenario x arm, 2000 draws, default_rng(20261104)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis import ARMS, PARENT_SCORES, SEED, expit, logit, loss, shuffle_within

HERE = Path(__file__).resolve().parent


def d_add_arm(agent, scen, arm, y):
    A, S = agent.max() + 1, scen.max() + 1
    k = np.zeros((A, S, 2)); n = np.zeros((A, S, 2))
    np.add.at(k, (agent, scen, arm), y); np.add.at(n, (agent, scen, arm), 1)
    # arm-specific
    k_sr, n_sr = k.sum(0), n.sum(0)                     # (S, 2)
    k_ar, n_ar = k.sum(1), n.sum(1)                     # (A, 2)
    K_r, N_r = k.sum((0, 1)), n.sum((0, 1))             # (2,)
    kc, nc = k[agent, scen, arm], n[agent, scen, arm]
    p_tr = (k_sr[scen, arm] - kc + 1) / (n_sr[scen, arm] - nc + 2)
    ko = k_ar[agent, arm] - kc; no = n_ar[agent, arm] - nc
    koth = (K_r[arm] - k_sr[scen, arm]) - ko; noth = (N_r[arm] - n_sr[scen, arm]) - no
    off_ar = logit((ko + 1) / (no + 2)) - logit((koth + 1) / (noth + 2))
    # pooled-arm offset (frozen D_add's off_a)
    kk, nn = k.sum(2), n.sum(2)
    kc2, nc2 = kk[agent, scen], nn[agent, scen]
    ko2 = kk.sum(1)[agent] - kc2; no2 = nn.sum(1)[agent] - nc2
    koth2 = (kk.sum() - kk.sum(0)[scen]) - ko2; noth2 = (nn.sum() - nn.sum(0)[scen]) - no2
    off_a = logit((ko2 + 1) / (no2 + 2)) - logit((koth2 + 1) / (noth2 + 2))
    lt = loss(p_tr, y)
    return float((lt - loss(expit(logit(p_tr) + off_ar), y)).mean()), float((lt - loss(expit(logit(p_tr) + off_a), y)).mean())


def main() -> None:
    rows = [json.loads(x) for x in (HERE / "results" / "scores.jsonl").read_text().splitlines()]
    models = sorted({r["model"] for r in rows}); tasks = sorted({r["task"] for r in rows})
    agent = np.array([models.index(r["model"]) for r in rows]); scen = np.array([tasks.index(r["task"]) for r in rows])
    arm = np.array([ARMS.index(r["cell"]) for r in rows]); y = np.array([float(r["key"]) for r in rows])
    par = [json.loads(x) for x in PARENT_SCORES.read_text().splitlines()]
    par = [r for r in par if r["cell"] in ARMS and r["seed"] < 40]
    e1 = {}
    for m in models:
        pn = sum(r["key"] for r in par if r["model"] == m and r["cell"] == "named")
        pr = sum(r["key"] for r in par if r["model"] == m and r["cell"] == "runbook")
        nn_ = sum(r["key"] for r in rows if r["model"] == m and r["cell"] == "named")
        nr = sum(r["key"] for r in rows if r["model"] == m and r["cell"] == "runbook")
        e1[m] = {"parent_named": pn, "parent_runbook": pr, "parent_n_per_arm": 40, "new_named": nn_, "new_runbook": nr,
                 "new_n_per_arm": int(((agent == models.index(m)) & (arm == 0)).sum()),
                 "same_sign": bool(np.sign(pr - pn) == np.sign(nr - nn_))}
    obs_arm, obs_pool = d_add_arm(agent, scen, arm, y)
    rng = np.random.default_rng(SEED)
    strata = scen * 2 + arm
    null = np.array([d_add_arm(shuffle_within(agent, strata, rng), scen, arm, y) for _ in range(2000)])
    out = {"label": "EXPLORATORY (not preregistered; Tove #14199)",
           "E1_sign_repeat": {"per_agent": e1, "same_sign_count": sum(v["same_sign"] for v in e1.values()), "n_agents": len(e1)},
           "E2": {"D_add_arm_specific_offset": obs_arm, "D_add_pooled_offset_on_arm_task_rate": obs_pool,
                  "gain_arm_specific_minus_pooled": obs_arm - obs_pool,
                  "null_q95": {"arm": float(np.quantile(null[:, 0], 0.95)), "pooled": float(np.quantile(null[:, 1], 0.95))},
                  "null_mean": {"arm": float(null[:, 0].mean()), "pooled": float(null[:, 1].mean())},
                  "null_p_ge_obs": {"arm": float((null[:, 0] >= obs_arm).mean()), "pooled": float((null[:, 1] >= obs_pool).mean())}}}
    (HERE / "results" / "exploratory.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["E1_sign_repeat"]["same_sign_count"]), json.dumps(out["E2"]))


if __name__ == "__main__":
    main()
