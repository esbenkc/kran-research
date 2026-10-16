"""skeptic2: independent extraction of every sampler / calibration number for the SLT foldout (G. Wang feedback).

Reads only committed code constants and raw data in ../../data. Writes sampler_facts.json next to this file.
    uv run python projects/grokked-llc-closed-form/skeptics/skeptic2/sampler_facts.py
"""
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PROJ = HERE.parent.parent
D = PROJ / "data"
out: dict = {}


def rows(p):
    return list(csv.DictReader(open(p)))


# ---------------------------------------------------------------- 1. frozen setting
cal = json.loads((D / "calibration.json").read_text())
out["frozen"] = cal
src = (PROJ / "qnet.py").read_text()
out["sgld_update_in_code"] = [l.strip() for l in src.splitlines() if "0.5 * eps" in l]
out["thin_default"] = int(re.search(r"thin=(\d+)", src).group(1))

# ---------------------------------------------------------------- 2. calibration grid + selection rule
G = json.loads((D / "calibration" / "calibration_grid_main.json").read_text())
F = json.loads((D / "calibration" / "calibration_frozen.json").read_text())
out["grid_main"] = dict(
    n_runs=len(G),
    n_cases=len({r["case"] for r in G}),
    **{k: sorted({r[k] for r in G}) for k in ["scale_w", "crel", "gamma", "nbeta", "steps", "burn"]},
    chains=sorted({len(r["chains"]) for r in G}),
    diverged=sum(r["diverged"] for r in G),
)
by = collections.defaultdict(list)
for r in G:
    by[(r["crel"], r["gamma"], r["nbeta"])].append(r)
rank = []
for k, rs in by.items():
    a = [abs(r["rel_err"]) if not r["diverged"] else np.inf for r in rs]
    a1 = [abs(r["rel_err"]) if not r["diverged"] else np.inf for r in rs if r["scale_w"] == 1.0]
    rank.append(dict(crel=k[0], gamma=k[1], nbeta=k[2], worst_all_scales=max(a), worst_scale1=max(a1),
                     mean_all=float(np.mean(a)), diverged=sum(r["diverged"] for r in rs)))
rank.sort(key=lambda d: d["worst_all_scales"])
out["grid_ranking_by_worst_abs_rel_err"] = rank
chosen = [d for d in rank if (d["crel"], d["gamma"], d["nbeta"]) == (cal["crel"], cal["gamma"], cal["nbeta"])][0]
out["chosen_rank_position_all_scales"] = rank.index(chosen) + 1
out["chosen_rank_position_scale1"] = sorted(rank, key=lambda d: d["worst_scale1"]).index(chosen) + 1
out["chosen_in_grid"] = chosen
# per-scale worst error of the chosen setting
out["chosen_worst_by_scale"] = {
    str(s): max(abs(r["rel_err"]) for r in by[(cal["crel"], cal["gamma"], cal["nbeta"])] if r["scale_w"] == s)
    for s in out["grid_main"]["scale_w"]}

# frozen validation table; is the teacher the same as in the grid (same truth + same h_max)?
tab = []
for f in F:
    g = [r for r in G if r["case"] == f["case"] and r["scale_w"] == 1.0][0]
    tab.append(dict(case=f["case"], kind=f["kind"], p=f["p"], K=f["K"], n=f["n"], truth=f["truth"],
                    paper=f["paper"], lambda_hat=f["lambda_hat"], ratio=f["lambda_hat"] / f["truth"],
                    rel_err=f["rel_err"], quad=f["quad"], eps=f["eps"], h_min=f["h_min"], h_max=f["h_max"],
                    chains=f["chains"], secs=f["secs"], same_teacher_as_grid=bool(
                        f["truth"] == g["truth"] and abs(f["h_max"] - g["h_max"]) < 1e-12 * g["h_max"])))
out["calibration_frozen_table"] = tab
out["calibration_worst_abs_rel_err"] = max(abs(t["rel_err"]) for t in tab)
out["calibration_eps_range"] = [min(t["eps"] for t in tab), max(t["eps"] for t in tab)]
out["calibration_total_secs"] = sum(t["secs"] for t in tab)

# ---------------------------------------------------------------- 3. measurement runs (llc.csv + p31 frac 0.7)
L = rows(D / "llc.csv")
L31 = rows(D / "llc_p31_frac07.csv")
allrows = [r for r in L if int(r["p"]) != 31] + L31
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan
for k in ["gamma", "nbeta", "crel", "chains", "steps", "burn"]:
    out.setdefault("llc_csv_constants", {})[k] = sorted({r[k] for r in L + L31})
grok = [r for r in allrows if r["phase"] == "grok" and r["dropped"] == "0"]
mem = [r for r in allrows if r["phase"] == "mem" and r["dropped"] == "0"]
out["n_rows"] = dict(llc_csv=len(L), p31_frac07=len(L31), used_grok=len(grok), used_mem=len(mem),
                     dropped=sum(r["dropped"] == "1" for r in L + L31))
out["eps_range_grok"] = [min(f(r, "eps") for r in grok), max(f(r, "eps") for r in grok)]
out["eps_range_mem_vonly"] = [min(f(r, "eps") for r in mem), max(f(r, "eps") for r in mem)] if mem else None
out["train_loss_after_polish_grok"] = [min(f(r, "train_loss") for r in grok), max(f(r, "train_loss") for r in grok)]
out["polish_rel_change_grok_max"] = max(f(r, "polish_rel_change") for r in grok)
out["polish_steps_range"] = [min(int(r["polish_steps"]) for r in allrows), max(int(r["polish_steps"]) for r in allrows)]
# inside hypothesis + settled
st = {(int(r["p"]), int(r["K"])): r for r in rows(D / "stationarity.csv") if int(r["p"]) != 31}
st.update({(int(r["p"]), int(r["K"])): r for r in rows(D / "stationarity_p31_frac07.csv")})
out["stationarity"] = [dict(p=k[0], K=k[1], seed=int(v["seed"]), frac=float(v["frac"]),
                            lam10k=float(v["lambda_hat_10k"]), last_block=float(v["block5"]),
                            rel=float(v["rel_last_vs_10k"]), settled=int(v["settled"])) for k, v in sorted(st.items())]
st_all = rows(D / "stationarity.csv") + rows(D / "stationarity_p31_frac07.csv")
out["stationarity_counts_all_rows_incl_p31_frac05"] = dict(n=len(st_all), settled=sum(int(r["settled"]) for r in st_all))
out["stationarity_counts_used"] = dict(n=len(st), settled=sum(int(v["settled"]) for v in st.values()))
cells = collections.defaultdict(list)
for r in grok:
    cells[(int(r["p"]), int(r["K"]))].append(r)
verdict = []
for k, rs in sorted(cells.items()):
    inside = all(r["inside_hypothesis"] == "1" for r in rs)
    settled = int(st[k]["settled"]) if k in st else None
    ratios = [f(r, "lambda_hat") / f(r, "lambda_paper") for r in rs]
    rr = [f(r, "lambda_hat") / f(r, "lambda_rank") for r in rs]
    verdict.append(dict(p=k[0], K=k[1], n=len(rs), inside=inside, settled=settled, ratio_mean=float(np.mean(ratios)),
                        ratio_median=float(np.median(ratios)), ratios=ratios, ratio_vs_halfrank=rr,
                        lambda_paper=[f(r, "lambda_paper") for r in rs], lambda_hat=[f(r, "lambda_hat") for r in rs],
                        se=[f(r, "lambda_se") for r in rs], eps=[f(r, "eps") for r in rs]))
out["grok_cells"] = verdict
used = [v for v in verdict if v["inside"] and v["settled"] == 1]
out["headline"] = dict(configs=len(used), nets=sum(v["n"] for v in used),
                       median_of_cell_means=float(np.median([v["ratio_mean"] for v in used])),
                       range=[min(v["ratio_mean"] for v in used), max(v["ratio_mean"] for v in used)],
                       in_band=sum(abs(v["ratio_mean"] - 1) <= cal["band"] for v in used),
                       se_rel_max=float(max(s / l for v in used for s, l in zip(v["se"], v["lambda_hat"]))),
                       eps_range=[min(min(v["eps"]) for v in used), max(max(v["eps"]) for v in used)],
                       halfrank_ratio_range=[min(min(v["ratio_vs_halfrank"]) for v in used),
                                             max(max(v["ratio_vs_halfrank"]) for v in used)])
x = np.log([np.mean(v["lambda_paper"]) for v in used])
y = np.log([np.mean(v["lambda_hat"]) for v in used])
out["headline"]["loglog_slope_cell_means"] = float(np.polyfit(x, y, 1)[0])

# ---------------------------------------------------------------- 4. compute: total gradient steps
n_full = len(grok) + len(mem)
pruned_extra = sum(1 for r in grok if r["K_eff"] != r["K"])
sgld_runs = len(grok) + pruned_extra + 2 * len(mem)          # grok: full (+pruned); mem: V-only + full
out["compute"] = dict(sgld_runs_per_ckpt_rule="grok: full net (+ pruned net if K_eff<K); mem: V-only + full net",
                      sgld_runs=sgld_runs, steps_per_run=int(cal["chains"]) * int(cal["steps"]),
                      grad_evals_measurement=sgld_runs * int(cal["chains"]) * int(cal["steps"]),
                      grad_evals_stationarity=len(st_all) * 2 * 40_000,
                      grad_evals_calibration_grid=sum(len(r["chains"]) * r["steps"] for r in G),
                      grad_evals_calibration_frozen=sum(len(r["chains"]) * r["steps"] for r in F))

# ---------------------------------------------------------------- 5. traces on disk
tr = sorted((D / "llc_traces").glob("*.npz")) + sorted((D / "llc_traces_p31_frac07").glob("*.npz"))
z = np.load(tr[0])
out["traces"] = dict(n_files=len(tr), example=tr[0].name, keys=list(z.keys()),
                     shapes={k: list(z[k].shape) for k in z.keys()})

# wall time from logs, if any timestamps
for lg in ["run_llc.log", "run_llc_p31_frac07.log", "stationarity.log"]:
    txt = (D / lg).read_text(errors="ignore")
    out.setdefault("log_heads", {})[lg] = txt.splitlines()[:2] + txt.splitlines()[-2:]

(HERE / "sampler_facts.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: out[k] for k in ["frozen", "grid_main", "chosen_in_grid", "chosen_rank_position_all_scales",
                                      "chosen_rank_position_scale1", "chosen_worst_by_scale",
                                      "calibration_worst_abs_rel_err", "calibration_eps_range", "n_rows",
                                      "eps_range_grok", "eps_range_mem_vonly", "train_loss_after_polish_grok",
                                      "stationarity_counts_all_rows_incl_p31_frac05", "stationarity_counts_used",
                                      "headline", "compute", "traces", "llc_csv_constants", "thin_default",
                                      "sgld_update_in_code", "log_heads"]}, indent=1, default=float))
print("TOP 8 grid settings:")
for d in rank[:8]:
    print(d)
print("FROZEN TABLE:")
for t in tab:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items() if k != "chains"})
print("STATIONARITY:")
for s in out["stationarity"]:
    print(s)
print("CELLS:")
for v in verdict:
    print(v["p"], v["K"], v["n"], v["inside"], v["settled"], round(v["ratio_mean"], 3), [round(a, 3) for a in v["ratios"]])
