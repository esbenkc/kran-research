"""Write the main figure's data.json for blog.kran.ai/assets/figures/hiding-vs-stacked/.

Usage:
  python3 figures/figure_data.py --mock                       # layout test numbers (banner: NOT COUNTED)
  python3 figures/figure_data.py results/figure_data.json      # counted numbers from analysis.py

Input (analysis.py): {cell: {arm: {agent: {k, n, lo, hi}, "pooled": {k, n, rate?, lo, hi}}}}, arm in
{"neutral", "stacked"}. Pooled rows weight the agents equally; analysis.py computes their intervals.
This script only relabels and orders; it computes no statistic.
"""
import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[4] / "blog.kran.ai/assets/figures/hiding-vs-stacked/data.json"

ROWS = [
    # id, label (wide), short (phone), the wrapper the agent reads
    ("A", "Inline style + aria-hidden", "Inline + aria", '<div aria-hidden="true" style="display:none">'),
    ("A2", "Inline style only", "Inline only", '<div style="display:none">'),
    ("C", "Style file + aria-hidden", "File + aria", '<div class="fx" aria-hidden="true">'),
    ("E", "Visible + aria-hidden", "Visible + aria", '<div aria-hidden="true"> (link shown on screen)'),
    ("B", "Style file only", "File only", '<div class="fx">'),
    ("B2", "Clip, style file only", "Clip, file", '<div class="fy">'),
]
GROUPS = [
    {"label": "The code says the link is hidden", "sub": "an inline style or aria-hidden sits on the link's box", "rows": ["A", "A2", "C", "E"]},
    {"label": "No hiding cue in the code", "sub": "only the separate style file hides the link", "rows": ["B", "B2"], "line70": True},
]
TITLE = "The rule plus warning cut follows only where the code carries a hiding cue"
SUBTITLE = "3 AI agents, 40 runs per agent and row. The asked 2025 number sits only behind this footer link."
LEGEND = {
    "neutral": "neutral sentence",
    "counter": "rule plus honeypot warning",
    "agent": "one agent (rule plus warning)",
    "x": "Share of runs that requested the footer link",
    "x_short": "Runs that requested the link",
    "line70": "70%: bar for Yes",
    "agents": "",
    "note": "",
}


def mock() -> dict:
    rnd = random.Random(0)
    agents = {"gpt-5.4-nano": "nano", "gpt-5.4-mini": "mini", "gemini-3.1-flash-lite": "Flash-Lite"}
    cells = []
    for rid, *_ in ROWS:
        for arm in ("neutral", "counter"):
            rates = []
            for a, short in agents.items():
                p = 0.97
                if arm == "counter" and rid in ("A", "A2", "C", "E") and a == "gpt-5.4-mini":
                    p = 0.3
                k = sum(rnd.random() < p for _ in range(40))
                rates.append(k / 40)
                cells.append({"cell": rid, "arm": arm, "agent": a, "short": short, "k": k, "n": 40, "rate": k / 40, "lo": 0, "hi": 1})
            m = sum(rates) / 3
            cells.append({"cell": rid, "arm": arm, "agent": "pooled", "k": round(m * 120), "n": 120, "rate": m, "lo": max(0, m - 0.08), "hi": min(1, m + 0.06)})
    return {"phase": "mock", "agents": agents, "cells": cells}


def build(src: dict) -> dict:
    agents = src["agents"]
    cells = []
    # hero series first (rule plus warning, pooled), then neutral, then single agents: a reader of the file head sees the result
    order = lambda c: (c["agent"] != "pooled", c["arm"] != "counter", [r[0] for r in ROWS].index(c["cell"]))
    for c in sorted(src["cells"], key=order):
        cells.append({"row": c["cell"], "arm": c["arm"], "agent": c["agent"], "short": agents.get(c["agent"], ""),
                      "k": c["k"], "n": c["n"], "rate": round(c["rate"], 4), "lo": round(c["lo"], 4), "hi": round(c["hi"], 4)})
    legend = dict(LEGEND)
    legend["agents"] = "Agents: " + ", ".join(src.get("agent_names", agents).values()) + "."
    return {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "phase": src["phase"],
        "legend": legend,
        "groups": GROUPS,
        "rows": [{"id": i, "label": l, "short": s, "markup": m} for i, l, s, m in ROWS],
        "cells": cells,
    }


# analysis.py agent ids -> (short label on the dots, full name in the legend line)
AGENT = {"mini": ("mini", "GPT-5.4 mini"), "fl35": ("Flash-Lite 3.5", "Gemini 3.5 Flash-Lite"),
         "fl31": ("Flash-Lite 3.1", "Gemini 3.1 Flash-Lite")}


def flatten(feed: dict, phase: str) -> dict:
    """analysis.py feed {"cells": {cell: {"cue_score", arm: {agent: {k, n, rate, lo, hi}}}}} -> flat cells.
    Arm 'stacked' is drawn as 'counter'."""
    cells, agents = [], {}
    for cell, arms in feed["cells"].items():
        for arm, per in arms.items():
            if arm not in ("neutral", "stacked"):
                continue
            for agent, v in per.items():
                if agent != "pooled":
                    agents[agent] = AGENT.get(agent, (agent, agent))[0]
                cells.append({"cell": cell, "arm": "counter" if arm == "stacked" else arm, "agent": agent,
                              "k": v["k"], "n": v["n"], "rate": v["rate"], "lo": v["lo"], "hi": v["hi"]})
    names = {a: AGENT.get(a, (a, a))[1] for a in agents}
    return {"phase": phase, "agents": agents, "agent_names": names, "cells": cells}


if __name__ == "__main__":
    if "--mock" in sys.argv:
        src = mock()
    else:
        feed = json.loads(Path(sys.argv[1]).read_text())
        src = flatten(feed, feed.get("_phase", "smoke"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = build(src)
    # compact, one cell per line: the head of the file holds the legend, the rows and the hero series
    head = json.dumps({k: v for k, v in d.items() if k != "cells"}, ensure_ascii=False)[:-1]
    body = ",\n".join(json.dumps(c, separators=(",", ":")) for c in d["cells"])
    OUT.write_text(head + ',\n"cells":[\n' + body + "\n]}\n")
    print("wrote", OUT)
