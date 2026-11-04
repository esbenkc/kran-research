"""Main figure data for residue-as-trait (Felix; Drew's task 8 taken over).

Reads results/scores.jsonl (status runs), results/summary.json (null q95), research/exploratory_rows.json (k-task
curve and the parent one-task screen, both EXPLORATORY). Writes
blog.kran.ai/assets/figures/residue-as-trait/data.json (path given as argv[1], default next to the blog clone).

Every row = '% smaller guessing error on a held-out task' = D / loss of the comparator guess, where D is the
PREREG s3 D_add (comparator = the other 4 agents' Beta(1,1) rate on the held-out task). The per-arm rows run the same
formula inside one route placement (cells of 3 runs, offsets from the other 7 tasks in that placement); they match
Vera's skeptic rebuild (named +0.0142, runbook +0.1850).
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = 0.020  # PREREG s3 Yes edge, nats per run


def logit(p):
    return math.log(p / (1 - p))


def expit(x):
    return 1 / (1 + math.exp(-x))


def loss(y, p):
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def d_add(rows):
    """PREREG s3 D_add over the given runs. Returns (D, mean loss of the task-rate guess)."""
    agents = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    k = {(a, t): 0 for a in agents for t in tasks}
    n = {(a, t): 0 for a in agents for t in tasks}
    for r in rows:
        k[(r["model"], r["task"])] += int(r["key"])
        n[(r["model"], r["task"])] += 1
    gain = base = 0.0
    for r in rows:
        a, s, y = r["model"], r["task"], int(r["key"])
        oth = [b for b in agents if b != a]
        tr = [t for t in tasks if t != s]
        p_t = (sum(k[(b, s)] for b in oth) + 1) / (sum(n[(b, s)] for b in oth) + 2)
        p_a = (sum(k[(a, t)] for t in tr) + 1) / (sum(n[(a, t)] for t in tr) + 2)
        p_o = (sum(k[(b, t)] for b in oth for t in tr) + 1) / (sum(n[(b, t)] for b in oth for t in tr) + 2)
        p_add = expit(logit(p_t) + logit(p_a) - logit(p_o))
        gain += loss(y, p_t) - loss(y, p_add)
        base += loss(y, p_t)
    return gain / len(rows), base / len(rows)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "blog.kran.ai/assets/figures/residue-as-trait/data.json"
    rows = [json.loads(l) for l in open(HERE / "results/scores.jsonl")]
    summ = json.load(open(HERE / "results/summary.json"))
    expl = json.load(open(HERE / "research/exploratory_rows.json"))

    d_all, b_all = d_add(rows)
    assert abs(d_all - summ["key"]["D_add"]) < 1e-9, (d_all, summ["key"]["D_add"])
    q95 = summ["key"]["null_q95"]["D_add"]
    arm = {c: d_add([r for r in rows if r["cell"] == c]) for c in ("runbook", "named")}

    def row(id_, group, label, d, base, planned, n, extra=None):
        r = {"id": id_, "group": group, "label": label, "d": round(d, 4), "base": round(base, 4),
             "pct": round(d / base, 4), "bar_pct": round(BAR / base, 4), "planned": planned, "n": n}
        r.update(extra or {})
        return r

    R = [row("all", "Fixed before the run", "All 240 runs, both route placements", d_all, b_all, True, 240,
             {"null_q95_pct": round(q95 / b_all, 4)})]
    R.append(row("runbook", "Split by where the route sat (not planned)", "Route in the runbook (120 runs)",
                 *arm["runbook"], False, 120))
    R.append(row("named", "Split by where the route sat (not planned)", "Route in the denial (120 runs)",
                 *arm["named"], False, 120))
    for kk in range(1, 8):
        c = expl["k_task_curve"][str(kk)]
        R.append({"id": f"k{kk}", "group": "How many other tasks the guess knows (not planned)",
                  "label": f"{kk} other task" + ("" if kk == 1 else "s") + (" (same as the fixed test)" if kk == 7 else ""), "d": c["gain_add_vs_task"],
                  "pct": c["pct_better_add_vs_task"], "planned": False, "k": kk})
    p1 = expl["parent_one_task_screen"]
    R.append({"id": "parent", "group": "How many other tasks the guess knows (not planned)",
              "label": "Yesterday's one task, 80 runs per agent", "d": p1["gain_parent_agent_vs_parent_pooled"],
              "pct": p1["pct_better"], "planned": False,
              "how": "yesterday's rate of each agent on the storage task vs yesterday's pooled rate, scored on today's 210 runs on the 7 new tasks"})

    data = {
        "title": "Knowing the agent cut the guessing error on a new task by 11%, mostly with the route in the runbook",
        "subtitle": "How much smaller the error of guessing an agent's key use on a task it has not done, when the guess "
                    "also knows how this agent behaved on other tasks. 0% = no better than the other agents' rate on that task.",
        "bar_label": "Yes bar fixed before the run",
        "note": "5 agents, 8 tasks, 3 runs per agent, task and route placement. Error = log loss. Rows marked 'not planned' "
                "were checks we added after the data came in.",
        "rows": R,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    for r in R:
        print(f"{r['id']:8s} d {r['d']:+.4f} pct {r['pct']:+.3f}" + (f" bar {r['bar_pct']:.3f}" if "bar_pct" in r else ""))


if __name__ == "__main__":
    main()
