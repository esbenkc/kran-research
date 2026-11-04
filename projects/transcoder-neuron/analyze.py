"""PREREG v2.6 status, Predictions A/B/C and all named rows, from data/pushes/L{L}{suffix}_{units,pushes}.csv.

  env/.venv/bin/python analyze.py            -> data/status.json   (full run, layers 8-11)
  env/.venv/bin/python analyze.py _smoke 8   -> data/status_smoke.json

R^2 is about the line y = x: 1 - sum (meas - pred)^2 / sum (meas - mean meas)^2, pooled over (unit, context)
points of a cell; 95% CI = bootstrap over units (2,000 draws, seed 0). Per-unit relative error = median over its
contexts of |meas - pred| / (|meas| + |pred|).
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

SUF = sys.argv[1] if len(sys.argv) > 1 else ""
LAYERS = [int(x) for x in sys.argv[2:]] or [8, 9, 10, 11]
NB = 2000


def r2(pred, meas):
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    den = ((meas - meas.mean()) ** 2).sum()
    return float(1 - ((meas - pred) ** 2).sum() / den) if den > 0 else float("nan")


def boot_r2(d, pc, mc, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby("unit")]
    P = [g[pc].values for g in groups]; M = [g[mc].values for g in groups]
    out = []
    for _ in range(NB):
        idx = rng.integers(0, len(groups), len(groups))
        out.append(r2(np.concatenate([P[i] for i in idx]), np.concatenate([M[i] for i in idx])))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def relerr(d, pc, mc):
    e = (d[mc] - d[pc]).abs() / (d[mc].abs() + d[pc].abs()).clip(lower=1e-12)
    return e.groupby(d["unit"]).median()


def verdict(rn, rr, defined):
    """per-layer rule (PREREG status rule); returns (verdict, NO clause)."""
    if not defined:
        return "UNDEFINED", None
    if rn >= 0.7 and rr <= 0.4:
        return "YES", None
    cl = []
    if rr >= 0.7:
        cl.append("REST>=0.7 (both faithful)")
    if rn < 0.4:
        cl.append("NEURON<0.4 (neuron copies fail too)")
    if rn - rr < 0.1:
        cl.append("gap<0.1 (no difference)")
    return ("NO", "; ".join(cl)) if cl else ("NEITHER", None)


PRIMARY = ("total_r", "meas_local_r")
ROWS = {  # name: (pred column, measured column) ; sign -1 means pred is negated
    "R2": PRIMARY,
    "R2_rnd_push": ("rnd_total_r", "rnd_meas_r"),
    "R2_fonly": ("fonly_r", "meas_local_r"),
    "R2_fonly_vs_tc": ("fonly_r", "total_r"),
    "R2_floor": ("floor_r", "meas_local_r"),
    "R2_jvp": ("jvp_r", "meas_local_r"),
    "R2_mirror": ("mir_total_r", "mir_meas_r"),
    "R2_graph_vs_e2e": ("total_g", "e2e_real"),
    "R2_graph_local": ("total_g", "meas_local_g"),
    "R2_tc_e2e": ("e2e_tc", "e2e_real"),
    "R2_defined_cell": ("meas_local_r", "e2e_real"),
    "R2_write_plus_graph": ("wr_pred", "wr_plus"),
    "R2_write_plus_real": ("wr_pred_r", "wr_plus"),
    "R2_write_rnd_plus_graph": ("wr_rnd_pred", "wr_rnd_plus"),
    "R2_write_rnd_plus_real": ("wr_rnd_pred_r", "wr_rnd_plus"),
}

res = {"suffix": SUF, "primary": "total_r vs meas_local_r (g_real), PREREG v2.6", "layers": {}, "cells": {},
       "law": [], "top5": {}}
allunits = []
for L in LAYERS:
    P = pd.read_csv(f"data/pushes/L{L}{SUF}_pushes.csv")
    U = pd.read_csv(f"data/pushes/L{L}{SUF}_units.csv")
    U["unit"] = np.arange(len(U))
    P = P[P.pre_ok > 0]
    P["neg_wr_pred"] = -P.wr_pred; P["neg_wr_pred_r"] = -P.wr_pred_r
    P["neg_wr_rnd_pred"] = -P.wr_rnd_pred; P["neg_wr_rnd_pred_r"] = -P.wr_rnd_pred_r
    lay = {}
    lay["defined_R2"] = r2(P.meas_local_r, P.e2e_real)
    lay["defined"] = bool(lay["defined_R2"] >= 0.7)
    lay["precision_maxabs_full_vs_cache"] = float(P.y_full_minus_cache.abs().max())
    lay["precision_1pct_median_e2e"] = float(0.01 * P.e2e_real.abs().median())
    lay["precision_pass"] = bool(lay["precision_maxabs_full_vs_cache"] < lay["precision_1pct_median_e2e"])
    lay["g_cos_median"] = float(P.g_cos.median()); lay["g_norm_ratio_median"] = float(P.g_norm_ratio.median())
    cells = {}
    for grp in ["NEURON", "GAP", "REST", "ORACLE"]:
        d = P[P.group == grp]
        if len(d) == 0:
            continue
        c = {"n_units": int(d.unit.nunique()), "n_pts": int(len(d))}
        for k, (pc, mc) in ROWS.items():
            c[k] = r2(d[pc], d[mc])
        c["R2_write_minus_graph"] = r2(d.neg_wr_pred, d.wr_minus)
        c["R2_write_minus_real"] = r2(d.neg_wr_pred_r, d.wr_minus)
        c["R2_write_rnd_minus_graph"] = r2(d.neg_wr_rnd_pred, d.wr_rnd_minus)
        c["R2_write_rnd_minus_real"] = r2(d.neg_wr_rnd_pred_r, d.wr_rnd_minus)
        c["R2_ci"] = boot_r2(d, *PRIMARY)
        c["R2_seed"] = [r2(d[d.seed == s][PRIMARY[0]], d[d.seed == s][PRIMARY[1]]) for s in sorted(d.seed.unique())]
        c["R2_theory"] = float(1 - d.tr2.sum() / d.r2.sum())
        c["R2_theory_rnd"] = float(1 - d.tr2_rnd.sum() / d.r2_rnd.sum())
        c["spearman"] = float(stats.spearmanr(d[PRIMARY[0]], d[PRIMARY[1]]).statistic)
        c["med_relerr"] = float(relerr(d, *PRIMARY).median())
        c["cos_vec_median"] = float(d.cos_vec.median())
        c["rank1_median"] = float(d.rank1.median())
        c["flips_median"] = float(d.flips.median()); c["flips_rnd_median"] = float(d.flips_rnd.median())
        c["curv_median"] = float(d.curv.median()); c["curv_rnd_median"] = float(d.curv_rnd.median())
        nf = d[d.flips <= (1 if grp != "ORACLE" else 0)]
        c["R2_noflip"] = r2(nf[PRIMARY[0]], nf[PRIMARY[1]]) if len(nf) > 10 else None
        c["n_noflip"] = int(len(nf))
        c["du_rel_median"] = float((d.du_norm / d.xn_norm).median())
        cells[grp] = c
    # sensitivity: NEURON at 0.7 both-sided = NEURON + gap-zone units with joint >= 0.7 (stratified gap draw, not reweighted)
    uj = U.set_index("unit").joint
    sens = P[(P.group == "NEURON") | ((P.group == "GAP") & (P.unit.map(uj) >= 0.7))]
    cells["NEURON_0.7"] = {"R2": r2(sens[PRIMARY[0]], sens[PRIMARY[1]]), "n_units": int(sens.unit.nunique()),
                           "note": "NEURON + gap-zone units with joint >= 0.7 (stratified draw, not reweighted)"}
    # status
    rn, rr = cells["NEURON"]["R2"], cells["REST"]["R2"]
    v, cl = verdict(rn, rr, lay["defined"])
    sv = [verdict(a, b, lay["defined"])[0] for a, b in zip(cells["NEURON"]["R2_seed"], cells["REST"]["R2_seed"])]
    lay["seed_verdicts"] = sv
    lay["pooled_verdict"] = v
    lay["verdict"] = v if (v in ("YES", "NO") and sum(s == v for s in sv) >= 2) or v not in ("YES", "NO") else "MIXED"
    lay["no_clause"] = cl
    # class-label shuffle null for the gap (NEURON vs REST units)
    rng = np.random.default_rng(L)
    dd = P[P.group.isin(["NEURON", "REST"])]
    ug = dd.groupby("unit").group.first()
    obs = rn - rr
    null = []
    for _ in range(1000):
        perm = pd.Series(rng.permutation(ug.values), index=ug.index)
        lab = dd.unit.map(perm)
        a, b = dd[lab == "NEURON"], dd[lab == "REST"]
        null.append(r2(a[PRIMARY[0]], a[PRIMARY[1]]) - r2(b[PRIMARY[0]], b[PRIMARY[1]]))
    null = np.array(null)
    lay["gap"] = float(obs)
    lay["gap_shuffle_p_two_sided"] = float((np.abs(null) >= abs(obs)).mean())
    lay["gap_shuffle_null_95"] = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
    # Kasper's ask: top-5 |pred - meas| points
    dd = P.assign(err=(P[PRIMARY[0]] - P[PRIMARY[1]]).abs()).sort_values("err", ascending=False).head(5)
    res["top5"][L] = dd[["group", "uid", "window", "t", "pre", "du_norm", "total_r", "meas_local_r", "e2e_real",
                         "flips", "curv"]].to_dict("records")
    res["layers"][L] = lay
    res["cells"][L] = cells
    # law: per-unit relative errors
    rr_read = relerr(P, *PRIMARY)
    rr_write = relerr(P, "wr_pred_r", "wr_plus")
    rr_write_g = relerr(P, "wr_pred", "wr_plus")
    Uu = U.set_index("unit")
    for uidx in rr_read.index:
        row = Uu.loc[uidx]
        if row.group == "ORACLE":
            continue
        allunits.append({"L": L, "group": row.group, "seed": int(row.seed), "uid": int(row.uid),
                         "joint": float(row.joint), "enc_cos": float(row.enc_cos), "dec_cos": float(row.dec_cos),
                         "relerr_read": float(rr_read[uidx]), "relerr_write": float(rr_write[uidx]),
                         "relerr_write_graph": float(rr_write_g[uidx]), "fire_rate": float(row.fire_rate),
                         "mean_act": float(row.mean_act), "e_norm": float(row.e_norm), "d_norm": float(row.d_norm)})
res["law"] = allunits


def partial_spearman(df, x, y, covs):
    R = df[[x, y] + covs].rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c].values for c in covs])
    for c in [x, y]:
        beta = np.linalg.lstsq(Z, R[c].values, rcond=None)[0]
        R[c + "_res"] = R[c].values - Z @ beta
    return float(np.corrcoef(R[x + "_res"], R[y + "_res"])[0, 1])


law = pd.DataFrame(allunits)
law["log_fire"] = np.log(law.fire_rate)
for L in LAYERS:
    law[f"is_L{L}"] = (law.L == L).astype(float)
covs = ["log_fire", "mean_act", "e_norm", "d_norm"] + [f"is_L{L}" for L in LAYERS[1:]]
predB = {"read_partial": partial_spearman(law, "joint", "relerr_read", covs),
         "write_partial_real_g": partial_spearman(law, "joint", "relerr_write", covs),
         "write_partial_graph_g": partial_spearman(law, "joint", "relerr_write_graph", covs),
         "read_raw_spearman": float(stats.spearmanr(law.joint, law.relerr_read).statistic),
         "per_layer_read_partial": {L: partial_spearman(law[law.L == L], "joint", "relerr_read",
                                                         ["log_fire", "mean_act", "e_norm", "d_norm"]) for L in LAYERS},
         "n_units": int(len(law))}
if predB["read_partial"] <= -0.2 and abs(predB["write_partial_real_g"]) < 0.1:
    predB["outcome"] = "HOLDS"
elif predB["read_partial"] > -0.1:
    predB["outcome"] = "FAILS"
else:
    predB["outcome"] = "UNSETTLED"
predB["write_g_note"] = "WRITE outcome uses g_real (v2.6 PRIMARY gradient); graph-g version reported beside it"
res["predB"] = predB

# status + Prediction A
vs = [res["layers"][L]["verdict"] for L in LAYERS]
res["status"] = "CONFIRMED" if vs.count("YES") >= 3 else "REJECTED" if vs.count("NO") >= 2 else "UNCONFIRMED"
res["predA"] = {"layers_yes": vs.count("YES"),
                "outcome": "held" if sum(res["cells"][L]["NEURON"]["R2"] >= 0.7 and res["cells"][L]["REST"]["R2"] <= 0.4
                                         for L in LAYERS) >= 3 else "failed"}
# Prediction C
cC = [(L, g, res["cells"][L][g]["R2"], res["cells"][L][g]["R2_theory"]) for L in LAYERS for g in ["NEURON", "GAP", "REST"]]
ok = sum(abs(a - b) <= 0.15 for _, _, a, b in cC)
res["predC"] = {"cells_within_0.15": ok, "n_cells": len(cC),
                "outcome": "HOLDS" if ok >= 9 else "FAILS" if ok <= 4 else "UNSETTLED",
                "cells": [{"L": L, "group": g, "R2": a, "R2_theory": b} for L, g, a, b in cC]}
# v1 (scout) on the WRITE arm: NEURON >= 0.7, REST <= 0.4 in >= 3 layers; read with the graph g (as written) and g_real
for gk in ["graph", "real"]:
    n = sum(res["cells"][L]["NEURON"][f"R2_write_plus_{gk}"] >= 0.7 and res["cells"][L]["REST"][f"R2_write_plus_{gk}"] <= 0.4
            for L in LAYERS)
    res[f"v1_write_{gk}"] = {"layers_gap": n, "outcome": "held" if n >= 3 else "failed"}
json.dump(res, open(f"data/status{SUF}.json", "w"), indent=1, default=float)
print("status", res["status"], "| predA", res["predA"], "| predB", {k: predB[k] for k in ["read_partial", "write_partial_real_g", "outcome"]},
      "| predC", res["predC"]["cells_within_0.15"], res["predC"]["outcome"])
for L in LAYERS:
    ly = res["layers"][L]
    print(f"L{L} defined {ly['defined_R2']:.3f} verdict {ly['verdict']} seeds {ly['seed_verdicts']} clause {ly['no_clause']} "
          f"gcos {ly['g_cos_median']:.2f} prec {ly['precision_pass']}")
    for g, c in res["cells"][L].items():
        if g == "NEURON_0.7":
            print(f"   {g:10s} R2 {c['R2']:.3f} n {c['n_units']}")
            continue
        print(f"   {g:7s} n={c['n_units']:3d} R2 {c['R2']:.3f} {np.round(c['R2_ci'],2)} seeds {np.round(c['R2_seed'],2)} theory {c['R2_theory']:.3f} "
              f"rnd {c['R2_rnd_push']:.3f} fonly {c['R2_fonly']:.3f} graph {c['R2_graph_vs_e2e']:.2f} wr_g {c['R2_write_plus_graph']:.2f} "
              f"wr_r {c['R2_write_plus_real']:.2f} sp {c['spearman']:.2f} relerr {c['med_relerr']:.2f} flips {c['flips_median']:.0f}/{c['flips_rnd_median']:.0f}")
