"""THE figure: share of units that pass the one-main-job audit, per arm and layer group.

Input: results.json (code/judge/aggregate.py). Two nested rates per arm: the pre-registered one-main-job rate
(fit >= 15/20 and other-family detection >= 4/5) and the looser rate that also counts related offshoots toward the
15 (closest to Cammarata's wording). Unit rate = mean of the two judges; CIs = 95% bootstrap over units.
The judges' job-TYPE split is not shown: the two judges agree on type for only 60% of units (results.json).
Writes figures/<FIG_ID>/{data.json, plot.js} and copies them into the blog clone in this work dir.
"""
from __future__ import annotations

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
METRICS = [("one_job", "One main job (pre-registered rule)"),
           ("rel", "One main job, counting related offshoots")]


def main(results=PROJ / "results.json"):
    """Nested bars: the strict rate (one_job) is a subset of the offshoot rate (rel) for every unit and judge."""
    res = json.load(open(results))
    panels = [("all", "All three layers")] + [(L, f"Layer {L}") for L in (4, 14, 24)]
    rows = []
    for pk, plabel in panels:
        for arm, alabel in ARMS:
            e = res["rates"][f"{arm}/L{pk}"]
            for k, mlabel in METRICS:
                m, lo, hi = e[k]
                rows.append(dict(panel=plabel, arm=alabel, metric=mlabel, mean=round(m, 4), lo=round(lo, 4),
                                 hi=round(hi, 4), n=e["n"]))
    data = dict(
        title="Share of units that pass a blind one-main-job audit",
        subtitle="Qwen3-0.6B MLP units, 50 per layer per arm; mean of two LLM judges; whiskers: 95% bootstrap CIs over units",
        panels=[p for _, p in panels], arms=[a for _, a in ARMS], metrics=[m for _, m in METRICS], rows=rows,
        # PREREGISTRATION.md: H4 trained neurons pass (any type) in [50%, 90%]; H3 random network passes >= 40%.
        # H1/H2 are gaps on non-lexical jobs and have no mark on this axis.
        predictions=[dict(arm=ARMS[0][1], y=0.5, label="Predicted: 50–90%"),
                     dict(arm=ARMS[2][1], y=0.4, label="Predicted: at least 40%")])
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
