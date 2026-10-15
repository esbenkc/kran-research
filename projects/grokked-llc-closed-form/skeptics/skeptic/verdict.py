"""skeptic: recompute the pre-registered verdict (plan.md) from raw llc.csv + stationarity.csv,
and again from my own rerun (rerun.jsonl) where it overlaps.

Rule: grokked cells inside the hypothesis K_eff(3p-2) < p(N-1) (recomputed here, not read),
settled at 40k steps, p31 frac-0.5 rows excluded (frac confound, unsettled).
CONFIRMED iff >=80% of cells have |lambda_hat/lambda_paper - 1| <= 0.15 AND log-log slope in [0.85,1.15].
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(HERE, "../../data")
rows = list(csv.DictReader(open(f"{DD}/llc.csv")))
st = {(r["p"], r["K"]): r for r in csv.DictReader(open(f"{DD}/stationarity.csv"))}
p31 = f"{DD}/llc_p31_frac07.csv"
rows07 = list(csv.DictReader(open(p31))) if os.path.exists(p31) else []

out = {}


def cells(rows, drop_p31_frac05=True, need_settled=True):
    by = {}
    for r in rows:
        if r["phase"] != "grok":
            continue
        p, K, N, Ke = int(r["p"]), int(r["K"]), int(r["N"]), int(float(r["K_eff"]))
        if drop_p31_frac05 and p == 31 and abs(float(r["frac"]) - 0.5) < 1e-9:
            continue
        inside = Ke * (3 * p - 2) < p * (N - 1)
        assert inside == (r.get("inside_hypothesis", str(int(inside))) in ("1", "True")), r
        if not inside:
            continue
        s = st.get((r["p"], r["K"]))
        if need_settled and (s is None or s["settled"] != "1"):
            continue
        paper = Ke * (3 * p - 1) / 2
        assert abs(paper - float(r.get("lambda_paper", paper))) < 1e-9
        by.setdefault((p, K), []).append((float(r["lambda_hat"]), paper, Ke * (3 * p - 2) / 2))
    return by


def verdict(by, tag):
    c = []
    for (p, K), v in sorted(by.items()):
        lh = np.mean([x[0] for x in v]); lp = np.mean([x[1] for x in v]); ls = np.mean([x[2] for x in v])
        c.append(dict(p=p, K=K, n=len(v), lam_hat=lh, paper=lp, ratio=lh / lp, ratio_shift=lh / ls,
                      seed_ratios=[x[0] / x[1] for x in v]))
    if not c:
        return None
    r = np.array([x["ratio"] for x in c])
    slope = np.polyfit(np.log([x["paper"] for x in c]), np.log([x["lam_hat"] for x in c]), 1)[0]
    in_band = float(np.mean(np.abs(r - 1) <= 0.15))
    res = dict(cells=c, n_cells=len(c), n_nets=int(sum(x["n"] for x in c)), median_ratio=float(np.median(r)),
               min_ratio=float(r.min()), max_ratio=float(r.max()), frac_in_band=in_band, slope=float(slope),
               max_ratio_vs_shift=float(max(x["ratio_shift"] for x in c)),
               median_by_p={str(p): float(np.median([x["ratio"] for x in c if x["p"] == p])) for p in sorted({x["p"] for x in c})},
               verdict="CONFIRMED" if (in_band >= 0.8 and 0.85 <= slope <= 1.15) else "NULL")
    print(tag, {k: v for k, v in res.items() if k != "cells"})
    for x in c:
        print("   ", x["p"], x["K"], x["n"], round(x["lam_hat"], 1), x["paper"], round(x["ratio"], 4), [round(s, 3) for s in x["seed_ratios"]])
    return res


out["main_p11_23_settled"] = verdict(cells(rows), "MAIN")
out["incl_p31_frac05_no_settle_req"] = verdict(cells(rows, False, False), "WITH p31 frac0.5")
if rows07:
    for r in rows07:
        r.setdefault("phase", "grok")
    # p31 frac0.7 rows: no stationarity row required here (reported separately)
    by = cells(rows, True, True)
    by07 = cells(rows07, False, False)
    by.update(by07)
    out["with_p31_frac07"] = verdict(by, "WITH p31 frac0.7 (as available)")

# my own rerun, same cells, p<=17 only
rr = [json.loads(l) for l in open(os.path.join(HERE, "rerun.jsonl"))]
mine = {}
for r in rr:
    if r["phase"] == "grok" and r["K_eff"] * (3 * r["p"] - 2) < r["p"] * (r["N"] - 1):
        mine.setdefault((r["p"], r["K"]), []).append((r["lam_hat"], r["lam_paper"], r["lam_shift"]))
out["mine_rerun_inside"] = verdict(mine, "MY RERUN (inside cells I reran)")

# my rerun vs build, ckpt by ckpt
cmp = []
for r in rr:
    b = [x for x in rows if int(x["p"]) == r["p"] and int(x["K"]) == r["K"] and r["name"].startswith(f"p{x['p']}_K{x['K']}_s{x['seed']}_{x['phase']}")]
    if b:
        cmp.append(dict(name=r["name"], mine=r["lam_hat"], build=float(b[0]["lambda_hat"]), rel=r["lam_hat"] / float(b[0]["lambda_hat"]) - 1))
out["mine_vs_build"] = cmp
rel = np.array([x["rel"] for x in cmp])
print("mine vs build: n", len(cmp), "max |rel|", float(np.abs(rel).max()), "median", float(np.median(rel)))
for x in cmp:
    print("   ", x["name"], round(x["mine"], 1), round(x["build"], 1), f"{x['rel']:+.3%}")
json.dump(out, open(os.path.join(HERE, "verdict.json"), "w"), indent=1)
