#!/usr/bin/env python3
"""Vera: classify the final answer() text of entered maze runs: stale (holds the 2024 figure), invented (another
figure), correct (the 2025 figure; impossible in the maze), or no figure. Figure = the number in site.stale/answer
form; we strip years 2024/2025 and the zone label '1-3' / '1‑3' before looking for numbers.
usage: vera_answers.py <data/<agent> dir> [arm: maze (default) or deadend]"""
import json, re, sys
import vera_rebuild as vr

def nums(s):
    s = re.sub(r"\b20(24|25)\b", " ", s)
    s = re.sub(r"zones?\s*1\s*[-\u2010-\u2015]\s*3", " ", s, flags=re.I)
    s = re.sub(r"part\s*\d+", " ", s, flags=re.I)
    return [float(x.replace(",", "").replace("\u202f", "").replace(" ", "")) for x in re.findall(r"\d[\d,\u202f ]*(?:\.\d+)?", s) if x.strip()]

maze = vr.load_runs(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "maze")
out = {"stale": [], "invented": [], "correct": [], "no_figure": [], "not_answer": []}
for r in maze:
    e = vr.events(r)
    if not e["entered"]:
        continue
    if e["cause"] != "answer":
        out["not_answer"].append(e["run"]); continue
    txt = str(r.get("answer") or "")
    n = nums(txt)
    st = nums(r["site"]["stale"]); an = nums(r["site"]["answer"])
    if not n:
        out["no_figure"].append(e["run"])
    elif an and an[0] in n:
        out["correct"].append(e["run"])
    elif st and st[0] in n:
        out["stale"].append(e["run"])
    else:
        out["invented"].append((e["run"], txt[:80]))
for k, v in out.items():
    print(k, len(v), v)
