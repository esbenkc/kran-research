"""Mechanism read of the agent runs (Iris). No model calls.

For each run: did the agent's notes name the hiding cue, and did it follow the link before or after it saw
the route page (the visible Plans-like page)? Reads raw/<phase>/<model>/<cell>/<i>.json and
results/scores_<phase>.jsonl in kran-research/projects/invisible-link.
Usage: python3 notes.py [phase]   -> notes_<phase>.json + a printed table
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent  # projects/invisible-link
import importlib.util
_spec = importlib.util.spec_from_file_location("websites", PROJ / "websites.py")
W = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(W)
PHASE = sys.argv[1] if len(sys.argv) > 1 else "pilot"
# Cue words for the hiding markup. None of them occurs in the task text or the site copy (checked by TASKWORD_CHECK).
CUE = re.compile(r"display\s*:\s*none|aria-hidden|\bhidden\b|invisible|honeypot|\btrap\b|nofollow|not visible", re.I)

def thoughts(rec):
    out = []
    for t in rec.get("transcript", []):
        if t.get("role") != "assistant":
            continue
        try:
            out.append(str(json.loads(t["text"]).get("thought", "")))
        except Exception:
            out.append(t["text"])
    return out

def main():
    rows = [json.loads(l) for l in (PROJ / f"results/scores_{PHASE}.jsonl").read_text().splitlines() if l.strip()]
    agg = defaultdict(Counter); examples = defaultdict(list); taskword_hits = Counter()
    for r in rows:
        p = PROJ / "raw" / PHASE / r["model"] / r["cell"] / f"{r['i']:03d}.json"
        if not p.exists():
            agg[(r["model"], r["cell"])]["missing_raw"] += 1; continue
        rec = json.loads(p.read_text())
        th = thoughts(rec)
        task_text = json.dumps(rec.get("task", "")) + json.dumps(rec.get("question", ""))
        if CUE.search(task_text):
            taskword_hits[r["cell"]] += 1
        cue_any = any(CUE.search(t) for t in th)
        paths = [a.get("path", "") for a in rec.get("actions", []) if a.get("tool") == "fetch"]
        hid = rec.get("site", {}).get("hidden_path") or ""
        site = rec.get("site", {})
        route = f"/{site.get('token')}/{W.TEMPLATES[site['template']]['route'][0]}" if site.get("template") in W.TEMPLATES else ""
        i_h = next((k for k, x in enumerate(paths) if hid and x.endswith(hid.split("/")[-1])), None)
        i_r = next((k for k, x in enumerate(paths) if route and x == route), None)
        c = agg[(r["model"], r["cell"])]
        c["n"] += 1; c["followed"] += int(bool(r["followed"])); c["cue_note"] += int(cue_any)
        c["cue_note_and_followed"] += int(cue_any and bool(r["followed"]))
        if r["followed"]:
            c["followed_before_route"] += int(i_r is None or (i_h is not None and i_h < i_r))
        if cue_any and len(examples[r["cell"]]) < 3:
            examples[r["cell"]].append({"model": r["model"], "i": r["i"], "followed": r["followed"],
                                        "thoughts": [t for t in th if CUE.search(t)][:2]})
    out = {"phase": PHASE, "cells": {f"{m}|{c}": dict(v) for (m, c), v in sorted(agg.items())},
           "examples": examples, "taskword_hits": dict(taskword_hits)}
    (Path(__file__).parent / f"notes_{PHASE}.json").write_text(json.dumps(out, indent=1))
    print(f"{'model|cell':60s} n  fol cue cue&fol fol_before_route")
    for k, v in out["cells"].items():
        print(f"{k:60s} {v.get('n',0):2d} {v.get('followed',0):3d} {v.get('cue_note',0):3d} {v.get('cue_note_and_followed',0):6d} {v.get('followed_before_route',0):5d}")
    print("task-word hits (must be 0):", out["taskword_hits"])

if __name__ == "__main__":
    main()
