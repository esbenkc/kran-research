#!/usr/bin/env python3
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results" / "scores_counted.jsonl"
ANALYSIS = ROOT / "results" / "analysis.json"
CUES = ROOT / "results" / "cue_mentions.json"
FIGDATA = Path(__file__).resolve().parents[4] / "blog.kran.ai" / "assets" / "figures" / "invisible-link" / "data.json"

agents = ["gpt-5.4-nano", "gpt-5.4-mini", "gemini-3.1-flash-lite"]
title_cells = {
    "stuck-md-hidden_fit",
    "stuck-raw-hidden_fit",
    "stuck-md-hidden_nonfit",
    "stuck-raw-hidden_nonfit",
}

rows = [json.loads(line) for line in SCORES.read_text().splitlines()]
title_all = [r for r in rows if r["model"] in agents and r["cell"] in title_cells]
title = [r for r in title_all if r["exposed"]]

counts = {}
rates = {}
for agent in agents:
    for kind, cells in [
        ("fit", {"stuck-md-hidden_fit", "stuck-raw-hidden_fit"}),
        ("off", {"stuck-md-hidden_nonfit", "stuck-raw-hidden_nonfit"}),
    ]:
        s = [r for r in title if r["model"] == agent and r["cell"] in cells]
        counts[(agent, kind)] = (sum(bool(r["followed"]) for r in s), len(s))
        rates[(agent, kind)] = counts[(agent, kind)][0] / counts[(agent, kind)][1]
fit = sum(rates[(a, "fit")] for a in agents) / len(agents)
off = sum(rates[(a, "off")] for a in agents) / len(agents)
gap = fit - off

clusters = {a: {} for a in agents}
for r in title:
    agent = r["model"]
    template = r["template"]
    paraphrase = r["paraphrase"]
    kind = "fit" if r["cell"].endswith("hidden_fit") else "off"
    cell = clusters[agent].setdefault(template, {}).setdefault(paraphrase, {}).setdefault(kind, [0, 0])
    cell[0] += int(bool(r["followed"]))
    cell[1] += 1

def gap_from_sample(sample=False):
    vals = []
    for agent in agents:
        totals = {"fit": [0, 0], "off": [0, 0]}
        for _template, paras in clusters[agent].items():
            keys = sorted(paras)
            chosen = [random.choice(keys) for _ in keys] if sample else keys
            for p in chosen:
                for kind, value in paras[p].items():
                    totals[kind][0] += value[0]
                    totals[kind][1] += value[1]
        vals.append(totals["fit"][0] / totals["fit"][1] - totals["off"][0] / totals["off"][1])
    return sum(vals) / len(vals)

random.seed(0)
draws = sorted(gap_from_sample(True) for _ in range(10000))
ci = (draws[250], draws[9749])

print("title_exposed", len(title), "of", len(title_all), "unexposed", len(title_all) - len(title))
for agent in agents:
    print(
        "agent",
        agent,
        "fit",
        f"{counts[(agent, 'fit')][0]}/{counts[(agent, 'fit')][1]}",
        f"{rates[(agent, 'fit')]:.4f}",
        "off",
        f"{counts[(agent, 'off')][0]}/{counts[(agent, 'off')][1]}",
        f"{rates[(agent, 'off')]:.4f}",
    )
print("equal_agent_fit", f"{fit:.4f}", "off", f"{off:.4f}", "gap", f"{gap:.4f}")
print("cluster_bootstrap_seed0_n10000_ci", f"[{ci[0]:.4f},{ci[1]:.4f}]")

for cell in sorted(title_cells):
    s = [r for r in title if r["cell"] == cell]
    print("cell", cell, "follow", f"{sum(bool(r['followed']) for r in s)}/{len(s)}")

for prefix in ["rfree", "rstuck"]:
    for kind in ["hidden_fit", "hidden_nonfit"]:
        s = [r for r in rows if r["model"] in agents and r["cell"].startswith(prefix) and r["cell"].endswith(kind) and r["exposed"]]
        print(prefix, kind, "follow", f"{sum(bool(r['followed']) for r in s)}/{len(s)}")

for cell in [
    "stuck-a11y-hidden_fit",
    "stuck-a11y-hidden_nonfit",
    "stuck-a11y-sronly_fit",
    "stuck-raw-visible_fit",
    "stuck-md-visible_fit",
    "stuck-raw-nolink",
]:
    s = [r for r in rows if r["model"] in agents and r["cell"] == cell]
    ex = [r for r in s if r["exposed"]]
    print(
        "control",
        cell,
        "follow_all",
        f"{sum(bool(r['followed']) for r in s)}/{len(s)}",
        "exposed",
        len(ex),
        "follow_exposed",
        f"{sum(bool(r['followed']) for r in ex)}/{len(ex)}",
    )

print(
    "first_after_home_fit",
    sum(1 for r in title if r["cell"].endswith("hidden_fit") and r["followed"] and r["follow_fetch_index"] == 2),
    "of",
    sum(1 for r in title if r["cell"].endswith("hidden_fit") and r["followed"]),
)
print("off_task_follows", sum(1 for r in title if r["cell"].endswith("hidden_nonfit") and r["followed"]))

cues = json.loads(CUES.read_text())
cue_keys = [k for k in cues["by_cell"] if k.endswith("|stuck-raw-hidden_fit")]
print("raw_fit_hidden_mentions", sum(cues["by_cell"][k]["named_at_or_before_follow"] for k in cue_keys), "of", sum(cues["by_cell"][k]["runs"] for k in cue_keys))

analysis = json.loads(ANALYSIS.read_text())
print("analysis_json_title", analysis["title"])

if FIGDATA.exists():
    fig = json.loads(FIGDATA.read_text())
    print("figure_answer", fig["answer"])
    for r in fig["rows"]:
        if r["need"] in {"stuck", "rstuck", "rfree"}:
            print("figure_row", r["need"], r["view"], r["arm"], f"{r['followed']}/{r['exposed']}", f"rate={r['rate']:.4f}")
