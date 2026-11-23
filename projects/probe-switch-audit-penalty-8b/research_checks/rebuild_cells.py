"""Bruno's stdlib rebuild of the per-run hide cell from raw per-prompt rows (no status.py import).

strict switch  = share of test prompts with whole-graph probed lo > 0 AND unprobed lo < 0 (bf16 lo; fp32 lo32; argmax top)
R_blind3       = 1 - mean(probed - unprobed under blind top-3 cut) / mean(probed - unprobed), cached tail, per_prompt_lo
hide cell      = strict >= 3/4 AND R_blind3 < 3/10 (G1 read from the audit's own field)
Run: python3 research_checks/rebuild_cells.py   (from the project folder)
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data")


def whole(run):
    w = json.loads((D / "whole" / f"{run}.json").read_text())
    by = {}
    for r in w["rows"]:
        if r["split"] == "test":
            by.setdefault(r["prompt"], {})[r["probe"]] = r
    n = len(by)
    s16 = sum(1 for v in by.values() if v[1]["lo"] > 0 and v[0]["lo"] < 0)
    s32 = sum(1 for v in by.values() if v[1]["lo32"] > 0 and v[0]["lo32"] < 0)
    am = sum(1 for v in by.values() if v[1]["top"] == v[1]["a_id"] and v[0]["top"] == v[0]["b_id"])
    return n, Fraction(s16, n), Fraction(s32, n), Fraction(am, n)


def r_blind3(run):
    a = json.loads((D / "audit" / f"{run}.json").read_text())
    p = a["per_prompt_lo"]
    base = sum(x - y for x, y in zip(p["probed"], p["unprobed"]))
    cut = sum(x - y for x, y in zip(p["blind_k3"], p["unprobed"]))
    return 1 - cut / base, a["headline"]["title_cell_blind_top3_removed"], a["headline"]["blind_top3_heads"]


runs = sorted(p.stem for p in (D / "audit").glob("*.json"))
for run in runs:
    n, s16, s32, am = whole(run)
    r3, r3_file, heads = r_blind3(run)
    hide = s16 >= Fraction(3, 4) and r3 < 0.3
    print(f"{run:8s} n={n} strict={float(s16):.4f} fp32={float(s32):.4f} argmax={float(am):.4f} "
          f"R_blind3={r3:.4f} (file {r3_file:.4f}) top3={heads} hide={hide}")
