"""THE figure: share of units that pass the one-main-job audit, per arm, stacked by job type.

Inputs: data/unit_table.csv (aggregate.py), results.json (aggregate.py).
Stack height for type t = mean over units of mean over the two judges of [one_job_j and type_j == t];
the three stacks sum to the prereg one-job rate. Whiskers = 95% bootstrap CI over units for the one-job
rate (top) and for the non-lexical one-job rate (H1/H2 quantity, at the top of the syntactic+semantic part).
Writes figures/<FIG_ID>/{data.json, plot.js} and copies them into the blog clone in this work dir.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
FIG_ID = "random-network-neuron-audit"
BLOG_FIG = PROJ.parents[2] / "blog.kran.ai" / "assets" / "figures"  # the work-dir clone, never ~/Desktop

ARMS = [("A", "Trained network, neurons"), ("C", "Trained network, random directions"), ("B", "Random network, neurons")]
TYPES = [("SEMANTIC", "Semantic (a topic or concept)"), ("SYNTACTIC", "Syntactic or positional"),
         ("LEXICAL", "Lexical (one token or string)")]


def main(table=PROJ / "data" / "unit_table.csv", results=PROJ / "results.json"):
    rows = list(csv.DictReader(open(table)))
    res = json.load(open(results))
    judges = res["judges"]
    panels = [("all", "All three layers")] + [(L, f"Layer {L}") for L in (4, 14, 24)]
    bars, whisk = [], []
    for pk, plabel in panels:
        for arm, alabel in ARMS:
            sel = [r for r in rows if r["arm"] == arm and (pk == "all" or int(r["layer"]) == pk)]
            if not sel:
                continue
            for t, tlabel in TYPES:
                v = sum(sum((r[f"one_job__{j}"] == "True") and r[f"type__{j}"] == t for j in judges) / len(judges)
                        for r in sel) / len(sel)
                bars.append(dict(panel=plabel, arm=alabel, type=tlabel, value=round(v, 4), n=len(sel)))
            e = res["rates"][f"{arm}/L{pk}"]
            for k, lab in [("one_job", "Passes the audit (any type)"), ("nonlex", "Passes, non-lexical job")]:
                m, lo, hi = e[k]
                whisk.append(dict(panel=plabel, arm=alabel, metric=lab, mean=round(m, 4), lo=round(lo, 4),
                                  hi=round(hi, 4), n=e["n"]))
    data = dict(
        title="Share of units that pass the one-main-job audit",
        subtitle="Qwen3-0.6B MLP layer units, two blind LLM judges, held-out detection check",
        panels=[p for _, p in panels], arms=[a for _, a in ARMS], types=[t for _, t in TYPES],
        bars=bars, whiskers=whisk)
    out = PROJ / "figures" / FIG_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1))
    shutil.copy2(HERE / "plot.js", out / "plot.js")
    if BLOG_FIG.exists():
        dest = BLOG_FIG / FIG_ID
        dest.mkdir(exist_ok=True)
        for f in ("data.json", "plot.js"):
            shutil.copy2(out / f, dest / f)
        print("published ->", dest)
    print("figure ->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
