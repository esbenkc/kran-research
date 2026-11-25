"""Build the post's main figure data (Owain): athanorscience.com/assets/figures/llc-bias-law/data.json from results.json.

    python3 figures/build_main.py [--out FILE]

Left panel: every primary cell on the five blind shapes, the law's prediction (pushed before the run) against the
measured SGLD error (3-seed mean). Right panel: share of those cells within 5 points, law L next to the nulls N1-N4
and the secondary law L_z. Numbers are read from results.json and re-derived here from its per-cell values.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BLOG = HERE.parents[2] / "athanorscience.com"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(BLOG / "assets/figures/llc-bias-law/data.json"))
args = ap.parse_args()

R = json.loads((HERE / "results.json").read_text())
assert R["prereg_commit"] == "84d8b1861"
prim = [c for c in R["cells"] if c["primary"]]
assert len(prim) == R["n_primary"] == 180, len(prim)
assert {c["group"] for c in prim} == {"blind"}


def miss(c, law):
    if not c["converged"] or not math.isfinite(c["e_meas"]):
        return 100.0
    return 100 * abs(c["pred"][law] - c["e_meas"])


def share(law):
    return sum(miss(c, law) <= 5 for c in prim) / len(prim)


# re-derive the frozen score from the per-cell values and check it against score.py
P = R["primary"]
assert abs(share("L") - P["L"]["share_within_5"]) < 1e-12
assert abs(st.median([miss(c, "L") for c in prim]) - P["L"]["median_miss"]) < 1e-9
for n in ("N1", "N2", "N3", "N4"):
    assert abs(share(n) - P["nulls"][n]["share_within_5"]) < 1e-12, n

SHAPE = {}
for c in prim:
    M, rest = c["case"].split("x", 1)
    N, rest = rest.split("x", 1)
    Hh, r = rest.split("r")
    SHAPE[c["case"]] = f"{M} → {Hh} → {N}, rank {r}"

LAB = {"L": "split law (the curve)", "L_z": "split law, shape-scaled (secondary)",
       "N1": "one constant", "N2": "one curve, no split", "N3": "the number we first planned",
       "N4": "lookup: same setting, training shape"}
bars = sorted(({"law": k, "label": LAB[k], "share": share(k), "median": st.median([miss(c, k) for c in prim]),
                "hero": k == "L"} for k in LAB), key=lambda d: d["median"])
need = 0.5 * min(P["nulls"][n]["median_miss"] for n in ("N1", "N2", "N3", "N4"))

cells = [{"case": c["case"], "shape": SHAPE[c["case"]], "nbeta": c["nbeta"], "q": c["q"], "c": c["c"],
          "gamma": c["gamma"], "pred": 100 * c["pred"]["L"],
          "meas": 100 * c["e_meas"] if math.isfinite(c["e_meas"]) else None, "ok": c["converged"],
          "miss": miss(c, "L"), "gibbs": None if c["e_gibbs"] is None else 100 * c["e_gibbs"]} for c in prim]

nL = sum(miss(c, "L") <= 5 for c in prim)
nuns = sum(not c["converged"] for c in prim)
out = {
    "status": P["status"], "n": len(prim), "n_within": nL, "n_unsettled": nuns,
    "share": share("L"), "median": P["L"]["median_miss"], "spread": P["spread_pts"],
    "seed_level": P["seed_level"], "need": need,
    "n2": json.loads((HERE / "predictions.json").read_text())["fit"]["N2"],
    "cells": cells, "bars": bars,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"{P['status']}: L {nL}/{len(prim)} within 5 ({share('L'):.3f}), not settled {nuns}; bars "
      + ", ".join(f"{b['law']} {b['share']:.2f}/{b['median']:.1f}" for b in bars))
print(args.out)
