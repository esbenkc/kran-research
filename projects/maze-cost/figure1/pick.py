"""Owain's frozen pick rule (room #24730, #24765), printed so a reviewer can check it.

agent    = the first status agent in analysis.json (else the argument)
maze run = observed median unique-page depth D over ENTERED maze runs of blocks 1-2 (round down),
           lowest run index i with that D and a complete raw/ + access/ pair
dead end = the same run index i in the dead-end arm (same template, paraphrase and site secret)

usage: python3 pick.py <project dir> [agent]
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

proj = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(proj))
import analysis  # noqa: E402  (Bruno's frozen scorer: score_run, load_agent)

import os; an = Path(os.environ.get("ANALYSIS", proj / "results" / "analysis.json"))
agent = sys.argv[2] if len(sys.argv) > 2 else json.loads(an.read_text())["status_agents"][0]
d = proj / "data" / agent
arms = analysis.load_agent(d, agent)
maze = [r for r in arms["maze"] if r["entered"] and r["block"] in (1, 2)]
depths = sorted(r["D"] for r in maze)
med = int(statistics.median(depths) // 1)
print(f"agent {agent}: entered maze runs in blocks 1-2: {len(maze)}")
print("depth counts:", dict(sorted(Counter(depths).items())), "median", statistics.median(depths), "->", med)
cands = sorted(r["run_index"] for r in maze if r["D"] == med
               and (d / "access" / "maze" / f"{r['run_index']:03d}.jsonl").exists()
               and (d / "raw" / "deadend" / f"{r['run_index']:03d}.json").exists())
print("candidates (run index):", cands)
i = cands[0]
print(f"PICK maze i={i:03d}, dead end i={i:03d}")
