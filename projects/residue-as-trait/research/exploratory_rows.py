"""EXPLORATORY rows (Iris, after the counted run; NOT in PREREG). Run from this folder: uv run --with scipy python
exploratory_rows.py  ->  exploratory_rows.json (0.2 s).

(1) k-task curve: how many OTHER tasks must an agent's rate come from before it predicts a task it never saw?
    For each held-out task t and every subset K of the other 7 tasks with |K| = k (k = 1..7, all subsets):
      agent rate p_a = Beta(1,1) key rate of the agent on K (both route placements pooled)
      base rate  p_b = Beta(1,1) pooled key rate of all 5 agents on K
      D_add form: p_t = Beta(1,1) rate on t of the other 4 agents; p_o = their Beta(1,1) rate on K;
                  p_add = expit(logit p_t + logit p_a - logit p_o)   (PREREG s3 formula with K in place of all 7)
    gain_agent_vs_base = mean over held-out runs of loss(p_b) - loss(p_a), nats per run
    gain_add_vs_task   = mean over held-out runs of loss(p_t) - loss(p_add), nats per run
    'pct_better' = gain divided by the comparator's mean loss.
    At k = 7 the two gains equal PREREG D_base and D_add (results/summary.json key.D_base, key.D_add).
(2) Yesterday's one-task screen: per-agent Beta(1,1) rate from the parent run (discoverable-route, storage task,
    route in denial or runbook, seeds 0-39, 80 runs per agent; vendored as parent_scores_named_runbook.jsonl from
    esbenkc/kran-research swarm/defense-1103-discoverable-route @ c288271e, projects/discoverable-route/results/
    scores.jsonl, cells named+runbook, seeds 0-39) as the prediction for today's 210 runs on the 7 NEW tasks
    (storage excluded), vs the parent's pooled rate. Plus Spearman of per-agent rates (n = 5).
"""
import itertools
import json
import math
from pathlib import Path

from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
R = [json.loads(l) for l in (HERE.parent / "results/scores.jsonl").read_text().splitlines()]
PAR = [json.loads(l) for l in (HERE / "parent_scores_named_runbook.jsonl").read_text().splitlines()]
assert len(R) == 240 and len(PAR) == 400
T = sorted({r["task"] for r in R})
M = sorted({r["model"] for r in R})
k = {(m, t): sum(r["key"] for r in R if r["model"] == m and r["task"] == t) for m in M for t in T}
n = {(m, t): sum(1 for r in R if r["model"] == m and r["task"] == t) for m in M for t in T}


def loss(kk, nn, p):
    return -(kk * math.log(p) + (nn - kk) * math.log(1 - p))


def logit(p):
    return math.log(p / (1 - p))


curve = {}
for size in range(1, 8):
    gb = ga = lb = lt = cnt = 0.0
    for t in T:
        others = [u for u in T if u != t]
        for K in itertools.combinations(others, size):
            for m in M:
                ka, na = sum(k[(m, u)] for u in K), sum(n[(m, u)] for u in K)
                kb, nb = sum(k[(x, u)] for x in M for u in K), sum(n[(x, u)] for x in M for u in K)
                pa, pb, po = (ka + 1) / (na + 2), (kb + 1) / (nb + 2), (kb - ka + 1) / (nb - na + 2)
                kt = sum(k[(x, t)] for x in M if x != m)
                nt = sum(n[(x, t)] for x in M if x != m)
                pt = (kt + 1) / (nt + 2)
                padd = 1 / (1 + math.exp(-(logit(pt) + logit(pa) - logit(po))))
                y, ny = k[(m, t)], n[(m, t)]
                gb += loss(y, ny, pb) - loss(y, ny, pa)
                ga += loss(y, ny, pt) - loss(y, ny, padd)
                lb += loss(y, ny, pb)
                lt += loss(y, ny, pt)
                cnt += ny
    curve[size] = {"gain_agent_vs_base": round(gb / cnt, 4), "pct_better_agent_vs_base": round(gb / lb, 4),
                   "gain_add_vs_task": round(ga / cnt, 4), "pct_better_add_vs_task": round(ga / lt, 4),
                   "n_subsets_per_task": math.comb(7, size)}

new = [r for r in R if r["task"] != "storage"]
pk = {m: sum(r["key"] for r in PAR if r["model"] == m) for m in M}
pn = {m: sum(1 for r in PAR if r["model"] == m) for m in M}
pr = {m: (pk[m] + 1) / (pn[m] + 2) for m in M}
p0 = (sum(pk.values()) + 1) / (len(PAR) + 2)
la = sum(loss(r["key"], 1, pr[r["model"]]) for r in new) / len(new)
l0 = sum(loss(r["key"], 1, p0) for r in new) / len(new)
nk = {m: sum(r["key"] for r in new if r["model"] == m) for m in M}
nn_ = {m: sum(1 for r in new if r["model"] == m) for m in M}
rho = spearmanr([pk[m] / pn[m] for m in M], [nk[m] / nn_[m] for m in M])
screen = {"per_agent": {m: {"parent_storage_80_runs": f"{pk[m]}/{pn[m]}", "today_storage": f"{k[(m, 'storage')]}/6",
                            "today_7_new_tasks": f"{nk[m]}/{nn_[m]}"} for m in M},
          "logloss_on_210_new_runs": {"parent_agent_rate": round(la, 4), "parent_pooled_rate": round(l0, 4)},
          "gain_parent_agent_vs_parent_pooled": round(l0 - la, 4),
          "pct_better": round((l0 - la) / l0, 4),
          "spearman_parent_vs_new_rates": {"rho": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 3),
                                           "n_agents": 5}}
out = {"label": "EXPLORATORY (not preregistered)", "k_task_curve": curve, "parent_one_task_screen": screen}
(HERE / "exploratory_rows.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
