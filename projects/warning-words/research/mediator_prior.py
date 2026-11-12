"""Iris, 2026-11-11: pre-data mediator check on warning-override (Nov 1) raw data, run BEFORE any warning-words data.
Within each sole-route (absent) cell: key use split by whether the window thought used a trap word
(`say_raw` from warning-override/says.py, honeypot-talk's frozen say rule).
Reads the parent's saved results straight from its git branch, so it needs no copies:
    python3 projects/warning-words/research/mediator_prior.py      (run inside the kran-research clone)
"""
import collections
import json
import subprocess

BRANCH = "origin/swarm/defense-1101-warning-displaces"
BASE = "projects/warning-override/results/"


def jsonl(name: str) -> list[dict]:
    txt = subprocess.run(["git", "show", f"{BRANCH}:{BASE}{name}"], capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


says = {(r["cell"], r["model"], r["seed"]): r for r in jsonl("says.jsonl")}
scores = {(r["cell"], r["model"], r["seed"]): r for r in jsonl("scores.jsonl")}
for cell in ["off_absent", "on_absent"]:
    tab = collections.defaultdict(lambda: [0, 0, 0, 0])
    for k, s in scores.items():
        if k[0] != cell:
            continue
        named = bool(says.get(k, {}).get("say_raw"))
        for m in (k[1], "ALL"):
            t = tab[m]
            t[0 if named else 2] += 1
            t[1 if named else 3] += s["key"]
    for m, t in sorted(tab.items()):
        print(cell, m[:24].ljust(24), "trap word: %d runs, %d used key | no trap word: %d runs, %d used key" % tuple(t))
