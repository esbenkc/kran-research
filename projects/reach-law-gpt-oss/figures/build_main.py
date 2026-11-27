"""Build the main figure's data.json from the analysis output (no hand-typed numbers).

Reads results/analysis.json (cells, Wilson intervals) and results/analysis_runs.csv (per-run rows, for the
three seed blocks of each figure row). Writes <blog clone>/assets/figures/reach-law-gpt-oss/data.json.

Usage: uv run python figures/build_main.py [--results results] [--blog ../../../athanorscience.com]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROWS = [
    # group, reach, label, short, selector over per-run rows
    ("pool", "reached", "Route reached the agent", "Reached", lambda r: r["phase"] == "main" and r["reached"]),
    ("pool", "never", "Route never reached it", "Never reached", lambda r: r["phase"] == "main" and not r["reached"]),
    ("arm", "never", "No route", "No route", lambda r: r["phase"] == "main" and r["arm"] == "none"),
    ("arm", "reached", "Runbook, route read", "Runbook, read", lambda r: r["phase"] == "main" and r["arm"] == "runbook" and r["reached"]),
    ("arm", "never", "Runbook, route not read", "Runbook, not read", lambda r: r["phase"] == "main" and r["arm"] == "runbook" and not r["reached"]),
    ("arm", "reached", "Note, route read", "Note, read", lambda r: r["phase"] == "main" and r["arm"] == "linked" and r["reached"]),
    ("arm", "never", "Note, route not read", "Note, not read", lambda r: r["phase"] == "main" and r["arm"] == "linked" and not r["reached"]),
    ("prompt", "reached", "Route in the task", "In the task", lambda r: r["phase"] == "prompt"),
]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--blog", default=str(ROOT.parents[2] / "athanorscience.com"))
    args = ap.parse_args()
    res = ROOT / args.results
    a = json.loads((res / "analysis.json").read_text())
    runs = []
    with (res / "analysis_runs.csv").open() as f:
        for r in csv.DictReader(f):
            r["reached"], r["key"], r["block"] = int(r["reached"]), int(r["key"]), int(r["block"])
            runs.append(r)

    rows = []
    for group, reach, label, short, sel in ROWS:
        rs = [r for r in runs if sel(r)]
        row = {"group": group, "reach": reach, "label": label, "short": short}
        if rs:
            k, n = sum(r["key"] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            seeds = []
            for b in range(3):
                rb = [r for r in rs if r["block"] == b]
                if rb:
                    seeds.append({"block": b, "k": sum(r["key"] for r in rb), "n": len(rb)})
            row.update(k=k, n=n, rate=k / n, lo=lo, hi=hi, seeds=seeds)
        else:
            row.update(k=None, n=0, empty="no runs")
        rows.append(row)

    # Cross-check against the analysis cells the status rests on.
    bins = a["bins"]
    assert (rows[0]["k"] or 0, rows[0]["n"]) == (bins["reached"]["k"], bins["reached"]["n"]), "reached bin mismatch"
    assert (rows[1]["k"] or 0, rows[1]["n"]) == (bins["never"]["k"], bins["never"]["n"]), "never bin mismatch"
    rb = a["runbook_reached"]
    assert (rows[3]["k"] or 0, rows[3]["n"]) == (rb["k"], rb["n"]), "runbook reached mismatch"

    n_arm = {arm: a["table"][arm]["all"]["n"] for arm in a["table"]}
    data = {
        "title": "XX head (set by Selma after the data)",
        "subtitle": f"gpt-oss-20b, {n_arm.get('none', 0)} / {n_arm.get('runbook', 0)} / {n_arm.get('linked', 0)} runs "
                    f"with no route / runbook / note, {a['prompt']['n']} with the route in the task",
        "legend": "Route: an allowed size report that finishes the task without the key. Reached: the route's text "
                  "appeared in what the agent had read before its first key call.",
        "xLabel": "share of runs that used the decoy key",
        "groups": [
            {"id": "pool", "label": "The test: no route, runbook and note groups pooled"},
            {"id": "arm", "label": "The same runs, split by where the route was written"},
            {"id": "prompt", "label": "Route written in the task (reported, not part of the test)"},
        ],
        "rows": rows,
        "status": a["status"],
    }
    out = Path(args.blog) / "assets" / "figures" / "reach-law-gpt-oss" / "data.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    for k in ("title", "subtitle", "legend", "groups", "note", "answer"):
        if k in old and not str(old[k]).startswith(("PENDING", "XX", "DUMMY")) and k != "subtitle":
            data[k] = old[k]  # keep text Selma set by hand; numbers always come from the analysis
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out}: {[(r['short'], r['k'], r['n']) for r in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
