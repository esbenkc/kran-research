"""Write the main figure's data.json for the post trap-aware-canary-open-weight.

Rows: last week's three API agents (parent run defense-1110, trap pages only, warning vs neutral sentence),
then the two open-weight agents on Esben's laptop queue, then Qwen3-1.7B on the mini (reported, never in the
answer). Marks: share of runs that requested the fitting link, by page (hidden = trap pages, visible = control
pages) and sentence (warning, neutral), with the 95% interval from the cells file.

Inputs (relative to the project dir):
  results/parent_api_cells.csv   parent cells, extracted once from the parent branch (see PARENT below)
  results/cells.csv              this run's cells, same columns as the parent's results/cells.csv,
                                 hiding in {cued, none|visible}, event 'fit', optional seed_lo, seed_hi
Output: figures/trap-aware-canary-open-weight/data.json (copy to the blog with --publish BLOG_DIR).

Usage: uv run python figures/figure_data.py [--phase counted|smoke] [--publish ../../../blog.kran.ai]
"""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
FIG_ID = "trap-aware-canary-open-weight"
PARENT = "af50d21f9:projects/trap-aware-canary/results/cells.csv"  # swarm/defense-1110-trap-aware-canary head

API = [("gpt-5.4-nano", "GPT-5.4 nano", "nano"), ("gpt-5.4-mini", "GPT-5.4 mini", "mini"),
       ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "Flash-Lite")]
LOCAL = [("gpt-oss-20b", "gpt-oss-20b", "gpt-oss-20b"), ("qwen3-8b", "Qwen3-8B", "Qwen3-8B")]
MINI = [("qwen3-1.7b", "Qwen3-1.7B", "Qwen3-1.7B")]


def parent_rows():
    path = PROJ / "results" / "parent_api_cells.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(subprocess.check_output(["git", "show", PARENT], cwd=PROJ, text=True))
    rows = list(csv.DictReader(path.open()))
    return [r for r in rows if r["carrier"] == "web" and r["block"] == "A" and r["hiding"] == "cued"
            and r["event"] == "fit" and r["sentence"] in ("warning", "neutral") and r["agent"] != "pooled"]


def new_rows():
    path = PROJ / "results" / "cells.csv"
    if not path.exists():
        return []
    return [r for r in csv.DictReader(path.open()) if r.get("event") == "fit" and r["sentence"] in ("warning", "neutral")
            and r["hiding"] in ("cued", "visible", "none") and r["agent"] != "pooled" and ":" not in r["cell"]]


def cell(r, name):
    c = {"agent": name, "page": "hidden" if r["hiding"] == "cued" else "visible", "sentence": r["sentence"],
         "k": int(r["k"]), "n": int(r["n"]), "rate": float(r["rate"]), "lo": float(r["ci_lo"]), "hi": float(r["ci_hi"])}
    if r.get("seed_lo"):
        c["seed_lo"], c["seed_hi"] = float(r["seed_lo"]), float(r["seed_hi"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="counted")
    ap.add_argument("--publish", default=None, help="blog clone dir; copies data.json into assets/figures/<id>/")
    a = ap.parse_args()

    cells, rows = [], []
    for key, name, short in API:
        rows.append({"agent": name, "short": short, "group": "api", "note": "no control pages in that test"})
        cells += [cell(r, name) for r in parent_rows() if r["agent"] == key]
    new = new_rows()
    for group, agents in (("local", LOCAL), ("mini", MINI)):
        for key, name, short in agents:
            mine = [cell(r, name) for r in new if r["agent"] == key]
            pending = not any(c["page"] == "hidden" for c in mine)
            rows.append({"agent": name, "short": short, "group": group, "pending": pending,
                         "note": "waiting for the laptop queue" if pending and group == "local" else
                                 ("not run" if pending else None)})
            cells += mine
    rows = [{k: v for k, v in r.items() if v is not None} for r in rows]

    data = {
        "title": "[FILL: result title for the figure head]",
        "subtitle": "Share of runs that requested the footer link that leads to the answer (our server log). "
                    "On trap pages the code hides the link from human readers; on control pages it shows.",
        "phase": a.phase,
        "legend": {
            "hidden_warned": "hidden link, trap warning",
            "hidden_neutral": "hidden link, neutral sentence",
            "visible_warned": "visible link, trap warning",
            "visible_neutral": "visible link, neutral sentence",
            "line50": "canary catches half",
            "x": "share of runs that requested the link (server log)",
            "x_short": "runs that requested the link",
            "note": "Bars: 95% intervals over runs. Last week: 60 runs per agent and sentence. "
                    "This test: 60 runs per trap cell over 3 seeds, 40 per control cell over 2 seeds.",
        },
        "groups": [
            {"id": "api", "label": "Last week: three small API agents"},
            {"id": "local", "label": "This test: two open models on a laptop"},
            {"id": "mini", "label": "Smaller check on our Mac mini (not part of the answer)"},
        ],
        "rows": rows,
        "cells": cells,
    }
    out = HERE / FIG_ID / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out} ({len(cells)} cells, {sum(1 for r in rows if r.get('pending'))} pending rows)")
    if a.publish:
        dst = Path(a.publish).resolve() / "assets" / "figures" / FIG_ID
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, dst / "data.json")
        print(f"published to {dst}")


if __name__ == "__main__":
    main()
