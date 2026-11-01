"""Build the main figure's data.json for the blog (assets/figures/warning-override/data.json).

Reads results/summary.json (written by analysis.py) only. Shares are model-equal means over the 5 agents,
the unit of the frozen status rule (PREREG s3). Run: python3 figure_data.py [out_path]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
S = json.loads((HERE / "results/summary.json").read_text())
MODELS = S["models"]
NAMES = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4-nano": "GPT-5.4 nano",
}


def cell(warn, flag):
    name = f"{'on' if warn else 'off'}_{'present' if flag else 'absent'}"
    rows = [S["cells"][f"{m}|{name}"] for m in MODELS]
    mean = lambda k: sum(r[k] / r["n"] for r in rows) / len(rows)
    for r in rows:  # the four route groups partition every run
        assert r["key_only"] + r["both_routes"] + r["flag_only"] + r["none"] == r["n"], (name, r)
    return {
        "flag": flag,
        "warn": warn,
        "n": sum(r["n"] for r in rows),
        "key_only": round(mean("key_only"), 4),
        "both": round(mean("both_routes"), 4),
        "flag_only": round(mean("flag_only"), 4),
        "agents": [
            {"model": NAMES[m], "any": round(r["any"] / r["n"], 4), "key": r["key"], "flag": r["flag"], "n": r["n"]}
            for m, r in zip(MODELS, rows)
        ],
    }


P = S["present"]
data = {
    "title": "The warning cut both routes: with the override flag on offer, flag use fell from 50% to 7%",
    "subtitle": "5 AI agents, 40 runs each per group (800 runs). Bar = share of runs, the 5 agents weighted equally.",
    "b": S["bound"],
    "cells": [cell(w, f) for f in (False, True) for w in (False, True)],
    "F": {k: round(v, 4) for k, v in zip(("point", "lo", "hi"), [P["F"]["point"], *P["F"]["ci90"]])},
    "K": {k: round(-v, 4) for k, v in zip(("point", "hi", "lo"), [P["K"]["point"], *P["K"]["ci90"]])},
    "note": "",
}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures/warning-override/data.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=1) + "\n")
for c in data["cells"]:
    print(c["flag"], c["warn"], c["key_only"], c["both"], c["flag_only"], round(c["key_only"] + c["both"] + c["flag_only"], 4))
print("F", data["F"], "K", data["K"])
