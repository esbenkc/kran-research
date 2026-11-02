"""Main figure data for the open-route post: results/summary.json -> blog assets/figures/open-route/data.json.

Usage: python figure_data.py [summary.json] [out data.json]
Every number on the figure comes from summary.json (all-five pooled rates with the joint bootstrap 95% interval,
the per-model counts, the 'key outside the allowed route' rates and the frozen cost-law predictions).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/summary.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[2] / "blog.kran.ai/assets/figures/open-route/data.json"

COLS = [  # (cell, column label, plain cost line, pooled key)
    ("floor", "Storage access allowed", "the approved token\ncovers storage", "p_floor"),
    ("open_c1", "Named route, 1 call", "the legal call costs\nthe same as the key", "p_open_c1"),
    ("open_c4", "Named route, 4 calls", "the legal route costs\n3 calls more", "p_open_c4"),
    ("blocked", "No route", "only the key\nreads the size", "p_blocked"),
]
OPEN = ("open_c1", "open_c4")
# The cost law is drawn on the 4-call group only: its 1-call prediction is the floor, and the floor group is
# split below because 45 of its 150 runs left out the token and got a denial (PREREG D4 row).


def pc(v):
    return f"{round(v * 100)}%"


def main():
    s = json.loads(SRC.read_text())
    a, o = s["all_five"], s["outside_route"]
    names = s["names"]
    cells = []
    for cid, label, sub, k in COLS:
        models = []
        for m in s["models"]:
            c = s["cells"][f"{m}|{cid}"]
            models.append({"model": names[m], "k": c["key"], "n": c["n"], "p": c["key"] / c["n"]})
        lo, hi = a["ci95"][k]
        cells.append({
            "id": cid, "label": label, "sub": sub, "p": a["point"][k], "lo": lo, "hi": hi,
            "n": sum(m["n"] for m in models),
            "pred": a["point"]["pred_4"] if cid == "open_c4" else None,
            "p_out": o["point"][k] if cid in OPEN else None,
            "models": models,
        })
    by = {c["id"]: c for c in cells}
    fs = s["first_storage_reply_split"]["floor"]
    by["floor"]["split"] = [
        {"text": "size on the first try", "k": fs["size"]["key"], "n": fs["size"]["n"]},
        {"text": "no token sent, denied", "k": fs["token_missing"]["key"], "n": fs["token_missing"]["n"]},
    ]
    # the law's 4-call value with the clean floor (runs whose first reply was the size), for the tooltip
    clean = fs["size"]["key"] / fs["size"]["n"]
    by["open_c4"]["pred_clean"] = clean + 0.75 * (a["point"]["p_blocked"] - clean)
    c4, c1, bl = by["open_c4"], by["open_c1"], by["blocked"]
    n_model = c4["models"][0]["n"]
    edge = pc(s["edge"])  # 10%
    verdict = {"CONFIRMED": f"It clears our {edge} bar.", "REJECTED": f"It stays below our {edge} bar."}.get(
        s["status"], f"Whether {pc(c4['p'])} clears our {edge} bar is not settled.")
    title = (f"Naming an allowed route cut key use from {pc(bl['p'])} to {pc(c1['p'])} (1 call) "
             f"and {pc(c4['p'])} (4 calls). {verdict}")
    data = {
        "title": title,
        "subtitle": f"5 AI agents, {n_model} runs each per group. Dot = share of runs, the 5 agents weighted equally.",
        # draw the status line only when the 4-call interval comes near it
        "edge": s["edge"] if c4["lo"] < s["edge"] + 0.1 else None,
        "cells": cells,
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(OUT)
    print(title)
    for c in cells:
        print(f"  {c['id']:8s} p={c['p']:.3f} [{c['lo']:.3f},{c['hi']:.3f}] pred={c['pred']} out={c['p_out']} "
              + " ".join(f"{m['k']}/{m['n']}" for m in c["models"]))


if __name__ == "__main__":
    main()
