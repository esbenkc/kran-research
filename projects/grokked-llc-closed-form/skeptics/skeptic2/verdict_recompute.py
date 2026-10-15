"""skeptic2: recompute the pre-registered verdict from raw data/llc.csv + data/stationarity.csv.

Independent of run.py / summary.json. numpy + csv only, single process.
Pre-registered rule (swarm plan.md): CONFIRMED if grokked lambda_hat / lambda_paper lies inside the
calibrated band (floor 15%) for >= 80% of inside-hypothesis (p,K) cells AND log-log slope in [0.85, 1.15];
NULL if outside the band with a consistent sign, or slope outside that range.

My own recomputation of every derived column (do not trust the csv's):
  lambda_paper = K_eff (3p-1)/2         (Cor 5.4 / Remark 5.5, v3)
  lambda_shift = K_eff (3p-2)/2         (shift-symmetry corrected count, skeptic2 derivation.md s2)
  inside       = K_eff (3p-2) < p (N-1) (one-hot subabundant condition, our condition)
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


llc = rows(os.path.join(DATA, "llc.csv"))
P31_07 = os.path.join(DATA, "llc_p31_frac07.csv")
if os.path.exists(P31_07):                      # lead's p31 retrain at train fraction 0.7
    llc += rows(P31_07)
stat = {(int(r["p"]), int(r["K"]), int(r["seed"]), float(r["frac"])): r
        for r in rows(os.path.join(DATA, "stationarity.csv"))}
cal = json.load(open(os.path.join(DATA, "calibration.json")))
band = max(0.15, cal["worst_calibration_rel_err"])

out = {"band": band, "cells": [], "checks": {}}
cells = defaultdict(list)
mismatch = []
for r in llc:
    if r["phase"] != "grok":
        continue
    p, K, s, frac, N = int(r["p"]), int(r["K"]), int(r["seed"]), float(r["frac"]), int(r["N"])
    Ke = int(r["K_eff"])
    lp = Ke * (3 * p - 1) / 2
    ls = Ke * (3 * p - 2) / 2
    inside = Ke * (3 * p - 2) < p * (N - 1)
    # cross-check csv's own derived columns
    if abs(lp - float(r["lambda_paper"])) > 1e-6 or inside != bool(int(r["inside_hypothesis"])):
        mismatch.append((p, K, s, lp, r["lambda_paper"], inside, r["inside_hypothesis"]))
    lh = float(r["lambda_hat"])
    st = stat.get((p, K, s, frac))
    cells[(p, K, frac)].append(dict(seed=s, N=N, K_eff=Ke, lam_hat=lh, lam_paper=lp, lam_shift=ls,
                                    inside=inside, rankJ=float(r["rankJ"]),
                                    lam_40k=float(st["lambda_hat_40k"]) if st else None,
                                    settled=int(st["settled"]) if st else None))

for (p, K, frac), rs in sorted(cells.items()):
    ratios = [x["lam_hat"] / x["lam_paper"] for x in rs]
    ratios_s = [x["lam_hat"] / x["lam_shift"] for x in rs]
    r40 = [x["lam_40k"] / x["lam_paper"] for x in rs if x["lam_40k"] is not None]
    settled = [x["settled"] for x in rs if x["settled"] is not None]
    out["cells"].append(dict(
        p=p, K=K, frac=frac, n_seeds=len(rs), inside=rs[0]["inside"], lam_paper=rs[0]["lam_paper"],
        lam_hat_mean=float(np.mean([x["lam_hat"] for x in rs])),
        ratio_min=min(ratios), ratio_max=max(ratios), ratio_mean=float(np.mean(ratios)),
        ratio_shift_max=max(ratios_s), ratio_40k=r40, settled_seed_checked=settled,
        rank_over_2_over_paper=float(np.mean([x["rankJ"] / 2 / x["lam_paper"] for x in rs]))))


def verdict(sel, label):
    if not sel:
        return {"label": label, "n": 0}
    inb = [abs(c["ratio_mean"] - 1) <= band for c in sel]
    x = np.log([c["lam_paper"] for c in sel])
    y = np.log([c["lam_hat_mean"] for c in sel])
    slope = float(np.polyfit(x, y, 1)[0]) if len(sel) > 1 else None
    signs = {np.sign(c["ratio_mean"] - 1) for c in sel}
    conf = (np.mean(inb) >= 0.8) and slope is not None and 0.85 <= slope <= 1.15
    return {"label": label, "n": len(sel), "cells": [f"p{c['p']}_K{c['K']}_f{c['frac']}" for c in sel],
            "ratio_means": [round(c["ratio_mean"], 3) for c in sel],
            "frac_in_band": float(np.mean(inb)), "loglog_slope": slope,
            "consistent_sign": len(signs) == 1, "all_seed_ratios_below_1-band":
                all(c["ratio_max"] < 1 - band for c in sel),
            "all_seed_ratios_vs_shift_below_1-band": all(c["ratio_shift_max"] < 1 - band for c in sel),
            "verdict": "CONFIRMED" if conf else "NULL"}


inside = [c for c in out["cells"] if c["inside"]]
main = [c for c in inside if c["p"] in (11, 17, 23)]          # lead #731: p31 frac .5 excluded
out["checks"]["csv_derived_column_mismatches"] = mismatch
out["checks"]["main_p11_23"] = verdict(main, "inside cells, p 11-23 (pre-registered set minus unsettled p31)")
out["checks"]["with_p31_frac05"] = verdict(inside, "inside cells incl. p31 frac 0.5 (robustness)")
p31_07 = [c for c in inside if c["p"] == 31 and abs(c["frac"] - 0.7) < 1e-9]
out["checks"]["main_plus_p31_frac07"] = verdict(main + p31_07, "inside cells p 11-23 + p31 frac 0.7")
out["checks"]["by_p_median_ratio"] = {
    p: float(np.median([c["ratio_mean"] for c in inside if c["p"] == p]))
    for p in sorted({c["p"] for c in inside})}

json.dump(out, open(os.path.join(HERE, "verdict_recompute.json"), "w"), indent=1, default=float)
for c in out["cells"]:
    print(f"p{c['p']:>2} K{c['K']:>3} f{c['frac']} n={c['n_seeds']} inside={int(c['inside'])} "
          f"paper={c['lam_paper']:7.1f} hat={c['lam_hat_mean']:7.1f} ratio {c['ratio_min']:.3f}-{c['ratio_max']:.3f} "
          f"vs_shift_max {c['ratio_shift_max']:.3f} 40k {[round(v, 3) for v in c['ratio_40k']]} "
          f"settled {c['settled_seed_checked']}")
print(json.dumps(out["checks"], indent=1, default=float))
