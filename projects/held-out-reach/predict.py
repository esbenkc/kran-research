"""Crawl -> cell class -> frozen predictions (PREREG v2.1). No model is called.

    .venv/bin/python predict.py [--crawl crawl] [--out predictions.json]

Cell class per tool x method (all 5 templates must agree, else the cell is 'mixed' and unscored):
  FREE = default view shows the hidden address; ROUTE = default hides it and one single action shows it;
  REACH0 = no single action shows it.
two-factor prediction = reach x s_a x u_a, with u_a = 1 on FREE cells and u_a from 1109 on ROUTE cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENTS = ["gpt-5.4-mini", "gpt-5.4-nano", "gemini-3.1-flash-lite"]
# PREREG v2.1 frozen terms. s_a: 1108 fit-link stuck runs (Vera rebuild #21154). cal: 1109 mcpfetch/display_none fit.
S = {"gpt-5.4-mini": (59, 60), "gpt-5.4-nano": (59, 60), "gemini-3.1-flash-lite": (60, 60)}
CAL = {"gpt-5.4-mini": (24, 30), "gpt-5.4-nano": (21, 30), "gemini-3.1-flash-lite": (9, 30)}
SCORED_ROUTE = {("lcpw", "display_none"), ("pwmcp", "display_none"), ("bufull", "display_none")}  # PREREG v2.2
SCORED_REACH0 = {("wbl", "display_none")}
FREE_CONTROLS = {("markitdown", "display_none"), ("pwmcp", "sr_only")}
REPLICATION = {("mcpfetch", "display_none")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", default="crawl,crawl_bufull")
    ap.add_argument("--out", default="predictions.json")
    a = ap.parse_args()
    rows = [json.loads(l) for c in a.crawl.split(",") for l in (ROOT / c / "reach.jsonl").read_text().splitlines() if l.strip()]
    by: dict = {}
    for r in rows:
        by.setdefault((r["tool"], r["method"]), []).append(r)
    cells = []
    for (tool, method), rs in sorted(by.items()):
        d = {r["default"] for r in rs}
        re_ = {r["reach"] for r in rs}
        if len(d) > 1 or len(re_) > 1:
            klass = "mixed"
        elif d == {1}:
            klass = "FREE"
        elif re_ == {1}:
            klass = "ROUTE"
        else:
            klass = "REACH0"
        role = ("scored_route" if (tool, method) in SCORED_ROUTE else "scored_reach0" if (tool, method) in SCORED_REACH0
                else "free_control" if (tool, method) in FREE_CONTROLS else "replication" if (tool, method) in REPLICATION
                else "crawl_only")
        for ag in AGENTS:
            s = S[ag][0] / S[ag][1]
            u = (CAL[ag][0] / CAL[ag][1]) / s
            reach = 1 if klass in ("FREE", "ROUTE") else 0
            two = reach * s * (1.0 if klass == "FREE" else u)
            cells.append({"tool": tool, "method": method, "agent": ag, "class": klass, "role": role,
                          "default": min(d), "reach": reach, "reach_actions": sorted({h.split(":", 1)[1] for r in rs for h in r["reach_actions"]}),
                          "s_a": round(s, 4), "u_a": round(u, 4), "pred_two_factor": round(two, 4),
                          "pred_one_factor": round(reach * s, 4), "pred_first_view": round(min(d) * s, 4),
                          "pred_const99": 0.99, "cal": CAL[ag]})
    body = {"prereg": "PREREG v2.1 (felix/PREREG_v2.md, 2026-11-12 22:58 PST)", "S": S, "CAL": CAL, "cells": cells}
    txt = json.dumps(body, indent=1, sort_keys=True)
    sha = hashlib.sha256(txt.encode()).hexdigest()
    (ROOT / a.out).write_text(txt + "\n")
    (ROOT / (a.out + ".sha256")).write_text(sha + "\n")
    for c in cells:
        if c["role"] != "crawl_only":
            print(f"{c['role']:14s} {c['tool']:10s} {c['method']:12s} {c['agent']:22s} {c['class']:6s} two={c['pred_two_factor']:.3f} one={c['pred_one_factor']:.3f}")
    print("sha256", sha)


if __name__ == "__main__":
    main()
