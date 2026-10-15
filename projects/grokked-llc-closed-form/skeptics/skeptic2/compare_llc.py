"""skeptic2: recompute the post's numbers from build's data/llc.csv with my own code, and compare
build's lambda_hat with my independent SGLD rerun (rerun_sgld.jsonl).

Prints: per-cell table, median lambda_hat/lambda_paper per p and phase (bias, and whether its sign
changes with p), log-log OLS slope of lambda_hat vs lambda_paper (grok, inside hypothesis), share of
inside cells within the 15% band, and build-vs-skeptic2 agreement per checkpoint.
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

LLC = sys.argv[1:] or ["../../kran-research/projects/grokked-llc-closed-form/data/llc.csv"]
rows = [r for path in LLC for r in csv.DictReader(open(path))]
rows = [r for r in rows if str(r.get("dropped", "")).lower() not in ("true", "1")]
f = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
print(f"{len(rows)} rows; columns: {list(rows[0].keys())}")

out = {}
for phase in sorted({r["phase"] for r in rows}):
    R = [r for r in rows if r["phase"] == phase]
    by_p = defaultdict(list)
    for r in R:
        ref = f(r, "lambda_paper")
        by_p[int(r["p"])].append(f(r, "lambda_hat") / ref)
    bias = {p: float(np.median(v)) for p, v in sorted(by_p.items())}
    print(f"\n[{phase}] median lambda_hat/lambda_paper by p:", {p: round(b, 3) for p, b in bias.items()})
    print(f"[{phase}] sign of (ratio-1) by p:", {p: int(np.sign(b - 1)) for p, b in bias.items()})
    out[phase] = dict(bias_by_p=bias)
    if phase.startswith("grok"):
        mine_flag = lambda r: f(r, "K_eff") * (3 * int(r["p"]) - 2) < int(r["p"]) * (int(r["N"]) - 1)
        theirs = lambda r: str(r.get("inside_hypothesis", "")).lower() in ("true", "1")
        mism = [(r["p"], r["K"], r["seed"]) for r in R if mine_flag(r) != theirs(r)]
        print(f"[{phase}] inside-hypothesis flag: my recomputation disagrees with llc.csv on {mism}")
        inside = [r for r in R if mine_flag(r)]
        x = np.log([f(r, "lambda_paper") for r in inside]); y = np.log([f(r, "lambda_hat") for r in inside])
        slope = float(np.polyfit(x, y, 1)[0]) if len(inside) > 2 else float("nan")
        within = float(np.mean([abs(f(r, "lambda_hat") / f(r, "lambda_paper") - 1) <= 0.15 for r in inside]))
        cells = sorted({(int(r["p"]), int(r["K"])) for r in inside})
        ib = defaultdict(list)
        for r in inside:
            ib[int(r["p"])].append(f(r, "lambda_hat") / f(r, "lambda_paper"))
        ibias = {p: (round(float(np.median(v)), 3), round(float(min(v)), 3), round(float(max(v)), 3)) for p, v in sorted(ib.items())}
        rq = [f(r, "lambda_hat") / f(r, "lambda_quad") for r in inside]
        print(f"[{phase}] inside cells n={len(cells)} {cells}; ckpts {len(inside)}; slope {slope:.3f}; within 15%: {within:.2f}")
        print(f"[{phase}] inside median/min/max ratio by p: {ibias}")
        print(f"[{phase}] inside lambda_hat/lambda_quad: median {np.median(rq):.3f} range {min(rq):.3f}-{max(rq):.3f}")
        out[phase].update(inside_cells=cells, slope_inside=slope, frac_within_band=within, inside_bias_by_p=ibias)

try:
    mine = {json.loads(l)["file"]: json.loads(l) for l in open("rerun_sgld.jsonl")}
    print("\nbuild vs skeptic2 SGLD (same frozen setting, independent code + polish):")
    agree = []
    for r in rows:
        key = f"p{r['p']}_K{r['K']}_s{r['seed']}_{r['phase'][:4]}.npz"
        if key in mine:
            a, b = f(r, "lambda_hat"), mine[key]["sgld_lambda"]
            agree.append(b / a - 1)
            print(f"  {key}: build {a:.1f}  skeptic2 {b:.1f}  diff {100*(b/a-1):+.1f}%  paper {f(r,'lambda_paper'):.1f}")
    out["rerun_rel_diff"] = agree
except FileNotFoundError:
    pass
json.dump(out, open("compare_llc.json", "w"), indent=1, default=str)
