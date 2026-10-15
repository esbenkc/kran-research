"""skeptic3: recompute the pre-registered verdict (plan.md) from raw data/llc.csv with my own code.

Rule (plan.md, pre-registered 05:58 PDT): CONFIRMED if grokked lambda_hat/lambda_paper is inside the calibrated band
(max(worst calibration rel. error, 0.15)) for >= 80% of inside-hypothesis (p,K) cells AND log-log slope of
lambda_hat vs lambda_paper in [0.85, 1.15]. NULL if outside the band with a consistent sign, or slope outside.

Also cross-checks build's lambda_hat against my own SGLD reruns (rerun_grok.jsonl, nbeta 1e4, unpolished ckpts).
Reads only CSV/JSON; light (no SGLD). Output: verdict_recompute.json
"""
import csv, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"

rows = list(csv.DictReader(open(DATA / "llc.csv")))
cal = json.load(open(DATA / "calibration.json"))
band = max(0.15, cal["worst_calibration_rel_err"])
stat = {(int(r["p"]), int(r["K"]), int(r["seed"])): r for r in csv.DictReader(open(DATA / "stationarity.csv"))}
phase_ok = {}
for r in csv.DictReader(open(DATA / "phase_labels.csv")):
    phase_ok[(int(r["p"]), int(r["K"]), int(r["seed"]), r["phase"])] = r


def f(x):
    return float(x) if x not in ("", "nan") else math.nan


# my own recomputation of the formula columns (do not trust the csv's lambda_paper)
for r in rows:
    p, Ke = int(r["p"]), int(r["K_eff"])
    if r["phase"] == "grok":
        mine = Ke * (3 * p - 1) / 2
    else:
        mine = p * min(int(r["l_rank"]), int(r["K"])) / 2
    assert abs(mine - f(r["lambda_paper"])) < 1e-9, (r["p"], r["K"], r["seed"], r["phase"], mine, r["lambda_paper"])
    inside = Ke * (3 * p - 2) < p * (int(r["N"]) - 1)
    if r["phase"] == "grok":
        assert inside == bool(int(r["inside_hypothesis"])), (p, Ke, r["N"], r["inside_hypothesis"])

out = {"band": band, "scenarios": {}}


def verdict(cells):
    ratios = {c: st.median(v) for c, v in cells.items()}
    xs = [math.log(c[2]) for c in cells]  # lambda_paper per cell (same K_eff across seeds? use median below)
    n = len(ratios)
    n_in = sum(abs(r - 1) <= band for r in ratios.values())
    # slope over seed-level points
    pts = [(math.log(lp), math.log(lh)) for c, v in cells.items() for lp, lh in c[3]]
    mx = st.mean(x for x, _ in pts); my = st.mean(y for _, y in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    signs = {r < 1 for r in ratios.values()}
    frac = n_in / n
    if frac >= 0.8 and 0.85 <= slope <= 1.15:
        v = "CONFIRMED"
    elif (frac < 0.8 and len(signs) == 1) or not (0.85 <= slope <= 1.15):
        v = "NULL"
    else:
        v = "UNCONFIRMED (mixed sign, band failed)"
    by_p = defaultdict(list)
    for c, r in ratios.items():
        by_p[c[0]].append(r)
    return {
        "n_cells": n, "n_in_band": n_in, "frac_in_band": frac, "loglog_slope_seed_level": slope,
        "all_below_1": signs == {True}, "verdict": v,
        "cell_ratio": {f"p{c[0]}_K{c[1]}": round(r, 4) for c, r in sorted(ratios.items())},
        "median_ratio_by_p": {p: round(st.median(v), 4) for p, v in sorted(by_p.items())},
        "min_ratio": min(ratios.values()), "max_ratio": max(ratios.values()),
    }


def grok_cells(exclude):
    cells = defaultdict(list)
    meta = {}
    for r in rows:
        if r["phase"] != "grok" or not int(r["inside_hypothesis"]) or int(r.get("dropped") or 0):
            continue
        p, K, s = int(r["p"]), int(r["K"]), int(r["seed"])
        if exclude(p, K, s, r):
            continue
        cells[(p, K)].append((f(r["lambda_paper"]), f(r["lambda_hat"])))
    out = {}
    for (p, K), v in cells.items():
        out[(p, K, st.median(x for x, _ in v), tuple(v))] = [y / x for x, y in v]
    return out


def unsettled(p, K, s, r):
    k = (p, K, s)
    return k in stat and not int(stat[k]["settled"])


out["scenarios"]["A_all_inside"] = verdict(grok_cells(lambda p, K, s, r: False))
out["scenarios"]["B_drop_p31_frac0.5"] = verdict(grok_cells(lambda p, K, s, r: p == 31 and f(r["frac"]) == 0.5))
out["scenarios"]["C_drop_p31_and_unsettled"] = verdict(
    grok_cells(lambda p, K, s, r: (p == 31 and f(r["frac"]) == 0.5) or unsettled(p, K, s, r)))
# D: only cells with a stationarity row that is settled (strictest)
out["scenarios"]["D_only_checked_settled"] = verdict(
    grok_cells(lambda p, K, s, r: not ((p, K, s) in stat and int(stat[(p, K, s)]["settled"]))))
# E: lead's final set (#860): p11-23 settled + p31 frac-0.7 cells that pass the 40k rule (stationarity_p31_frac07.csv)
rows_main = rows
p31 = list(csv.DictReader(open(DATA / "llc_p31_frac07.csv")))
st31 = {(int(r["p"]), int(r["K"])): int(r["settled"]) for r in csv.DictReader(open(DATA / "stationarity_p31_frac07.csv"))}
for r in p31:
    p, Ke = int(r["p"]), int(r["K_eff"])
    assert abs(Ke * (3 * p - 1) / 2 - f(r["lambda_paper"])) < 1e-9
rows = rows_main + p31
out["scenarios"]["E_lead_final_p31frac07_settled"] = verdict(grok_cells(
    lambda p, K, s, r: (p == 31 and f(r["frac"]) == 0.5) or (p == 31 and not st31.get((p, K), 0)) or unsettled(p, K, s, r)))
out["scenarios"]["F_p31frac07_both_K"] = verdict(grok_cells(
    lambda p, K, s, r: (p == 31 and f(r["frac"]) == 0.5) or unsettled(p, K, s, r)))
out["p31_frac07_40k"] = {f"p31_K{k[1]}": {"settled": v} for k, v in st31.items()}
rows = rows_main

# memorising (V-only) cells, phase label must pass under both predictors
mem = defaultdict(list)
for r in rows:
    if r["phase"] != "mem" or int(r.get("dropped") or 0):
        continue
    k = (int(r["p"]), int(r["K"]), int(r["seed"]), "mem")
    if k in phase_ok and not int(phase_ok[k]["mem_both"]):
        continue
    mem[(k[0], k[1])].append(f(r["lambda_hat"]) / f(r["lambda_paper"]))
out["mem_vonly_ratio"] = {f"p{p}_K{K}": [round(x, 4) for x in v] for (p, K), v in sorted(mem.items())}
allm = [x for v in mem.values() for x in v]
out["mem_vonly_summary"] = {"n": len(allm), "median": st.median(allm), "min": min(allm), "max": max(allm)}

# cross-check build vs my own SGLD (rerun_grok.jsonl, nbeta 1e4)
mine = defaultdict(list)
for fn in ["rerun_grok.jsonl", "rerun_grok_b.jsonl", "rerun_grok_p31.jsonl", "rerun_grok_p23.jsonl"]:
    if not (HERE / fn).exists():
        continue
    for line in open(HERE / fn):
        d = json.loads(line)
        for key, s in d.get("sgld", {}).items():
            if "_nb10000_" in key:
                mine[(d["p"], d["K"], int(d["file"].split("_s")[1].split("_")[0]))].append(s["lambda_hat"])
xc = {}
for r in rows:
    if r["phase"] != "grok":
        continue
    k = (int(r["p"]), int(r["K"]), int(r["seed"]))
    if k in mine:
        m = st.mean(mine[k])
        xc[f"p{k[0]}_K{k[1]}_s{k[2]}"] = {"build": round(f(r["lambda_hat"]), 1), "skeptic3": round(m, 1),
                                          "rel": round(m / f(r["lambda_hat"]) - 1, 4)}
out["crosscheck_build_vs_skeptic3_nbeta1e4"] = xc
out["stationarity_rows"] = {f"p{k[0]}_K{k[1]}_s{k[2]}": {"rel_40k_vs_10k": round(f(v["rel_last_vs_10k"]), 4),
                                                         "settled": int(v["settled"])} for k, v in stat.items()}
json.dump(out, open(HERE / "verdict_recompute.json", "w"), indent=1, default=str)
for k, v in out["scenarios"].items():
    print(k, v["verdict"], f"{v['n_in_band']}/{v['n_cells']} in band", f"slope {v['loglog_slope_seed_level']:.3f}",
          v["median_ratio_by_p"])
print("mem", out["mem_vonly_summary"])
print("xcheck", xc)
