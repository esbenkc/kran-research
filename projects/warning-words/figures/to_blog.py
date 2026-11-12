"""Writes the blog figure bundle data.json (blog.kran.ai/assets/figures/warning-words/data.json) from
results/figure_data.json (analysis.py). Labels and legend text live here; every number comes from the results file.

Usage: python figures/to_blog.py [path/to/blog.kran.ai]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BLOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "blog.kran.ai"
src = json.loads((HERE / "results" / "figure_data.json").read_text())

# display order, top to bottom: baseline, replication of the one-liner, the 2x2, then the content-free naming arm
ORDER = ["neutral", "trap", "honeypot", "nameit", "full", "verbalise"]
LABEL = {
    "neutral": "Neutral sentences\n(no warning)",
    "trap": "One-line\ntrap warning",
    "honeypot": "Honeypot\nsentence alone",
    "nameit": "Naming instruction\nalone",
    "full": "Both sentences\n(full warning)",
    "verbalise": "Naming with no\nhoneypot word",
}
SHORT = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4-nano": "GPT-5.4 nano",
}


def cells(panel, per_agent, pooled):
    out = []
    for c in per_agent:
        out.append({"panel": panel, "arm": c["arm"], "agent": SHORT.get(c["agent"], c["agent"]), "n": c["n"], "k": c["k"],
                    "rate": c["k"] / c["n"]})
    for p in pooled:
        mine = [c for c in per_agent if c["arm"] == p["arm"]]
        out.append({"panel": panel, "arm": p["arm"], "agent": "pooled", "n": sum(c["n"] for c in mine),
                    "k": sum(c["k"] for c in mine), "rate": p["rate"], "lo": p["lo"], "hi": p["hi"]})
    return out


key_agents = sorted({c["agent"] for c in src["cells"]})
web_agents = sorted({c["agent"] for c in src.get("web", [])})
n_key = sorted({c["n"] for c in src["cells"]})
winner = src.get("web_winner", "nameit")
data = {
    "title": "Each warning sentence alone cut key use by about 30 points. Together they cut 62.",
    "subtitle": "Share of runs that used the decoy key, by the sentences added to the agent's system prompt.",
    "phase": src.get("phase", "counted"),
    "mark": ["honeypot", "nameit"],
    "n_agents": {"key": len(key_agents), "web": len(web_agents)},
    "arms": [{"id": a, "label": LABEL[a]} for a in ORDER if any(c["arm"] == a for c in src["cells"])],
    "web_arms": [{"id": "neutral", "label": LABEL["neutral"]}, {"id": winner, "label": LABEL[winner]}],
    "legend": {
        "pool": f"{len(key_agents)} agents, equal weight",
        "agent": "one agent",
        "neutral_line": "neutral rate",
        "mark": "the two sentences, each alone",
        "xkey": "share of runs that used the decoy key (action log)",
        "xkey_short": "runs that used the key",
        "web": "Second test: a link hidden from human view",
        "web_sub": f"Share of runs that followed the hidden link (server log), {len(web_agents)} agents.",
        "xweb": "share of runs that followed the hidden link",
        "xweb_short": "runs that followed the link",
        "agents": "Agents: " + ", ".join(SHORT.get(a, a) for a in key_agents) + ". "
                  + f"{'/'.join(str(n) for n in n_key)} runs per agent and row. Website: "
                  + ", ".join(SHORT.get(a, a) for a in web_agents) + f", {'/'.join(str(n) for n in sorted({c['n'] for c in src.get('web', [])}))} runs each.",
        "note": "Bar: 95% interval. Scored from the action log and the server log; no model judged.",
    },
    "cells": cells("key", src["cells"], src["pooled"]) + cells("web", src.get("web", []), src.get("web_pooled", [])),
}
out = BLOG / "assets" / "figures" / "warning-words" / "data.json"
out.write_text(json.dumps(data, indent=1) + "\n")
print(f"wrote {out}: {len(data['cells'])} cells, phase {data['phase']}")
