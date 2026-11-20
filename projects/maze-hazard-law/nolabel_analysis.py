"""DEVIATIONS D8 (reported row only, never a status input): does gpt-oss-20b's rising leave chance survive when the
maze pages carry no page number? Compares the counted maze arm (data/gpt-oss-20b-low-full, raw/maze) with the
no-number arm (data/gpt-oss-20b-low-nolabel, raw/maze_nolabel). Same scoring and law code as analysis.py.

Reading rule, frozen in DEVIATIONS D8 before the job ran (Felix #26820 on Tove #26816; Vera #26822, #26830; Iris #26832):
  no rise detected  iff the no-number arm's Weibull shape k has a 95% profile interval that includes 1 AND its upper
                    end is below the labelled arm's pooled interval lower end (2.48, results/analysis.json)
  rise survives     iff that interval lies above 1 (lower end > 1)
  falling           iff that interval lies below 1 (upper end < 1)
  unclear           otherwise, and whenever there are fewer than 10 give-ups or no Weibull fit
Reported next to it: the cap share and the Kaplan-Meier median pages (labelled arm: 12). The arms differ in the page
number only; transcript length grows with depth in both, so "rise survives" cannot tell depth from transcript length
(Kasper #26808 (3)).

    .venv/bin/python nolabel_analysis.py      -> results/d8/nolabel.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import analysis as A

ROOT = Path(__file__).resolve().parent
ARMS = {"maze (page number shown)": ("gpt-oss-20b-low-full", "maze"),
        "maze_nolabel (no page number)": ("gpt-oss-20b-low-nolabel", "maze_nolabel")}
LABELLED_K_CI_LO = 2.48  # pooled profile 95% lower end of the labelled arm's k (analysis.json; Vera #26822)


def per_page(runs: list[dict]) -> list[dict]:
    n_at, ev = A.risk_tables(runs)
    return [{"page": d, "at_risk": int(n_at[d]), "giveups": int(ev[d])} for d in range(1, len(n_at)) if n_at[d] > 0]


def reading(L: dict) -> str:
    w = L["weibull"]
    if L["G"] < 10 or not w.get("k_ci"):
        return "unclear"
    lo, hi = w["k_ci"]
    if lo <= 1 <= hi and hi < LABELLED_K_CI_LO:
        return "no rise detected"
    if lo > 1:
        return "rise survives"
    if hi < 1:
        return "falling"
    return "unclear"


def main() -> None:
    out, ent_by_arm = {}, {}
    for label, (folder, arm) in ARMS.items():
        d = ROOT / "data" / folder
        if not (d / "raw" / arm).is_dir():
            out[label] = {"missing": str(d / "raw" / arm)}
            continue
        rows = A.load_agent(d)[arm]
        ent = [r for r in rows if r["entered"]]
        ent_by_arm[label] = ent
        L = A.law(ent)
        out[label] = {
            "folder": folder, "arm": arm, "n_runs": len(rows), "n_entered": len(ent),
            "blocks_entered": {int(b): sum(r["block"] == b for r in ent) for b in sorted({r["block"] for r in ent})},
            "causes": {c: sum(r["cause"] == c for r in ent) for c in sorted({r["cause"] for r in ent})},
            "G3": L["G3"], "E3": L["E3"], "h3": L["h3"], "G": L["G"], "h_all": L["h_all"],
            "km_median": L["km_median"], "pred_median": L["pred_median"], "R": L["R"],
            "cap_share": sum(r["cause"] == "cap" for r in ent) / len(ent) if ent else None,
            "pearson_p": L["chi"]["p"], "weibull_k": L["weibull"]["k"], "weibull_k_ci": L["weibull"]["k_ci"],
            "weibull_p": L["weibull"]["p"], "verdict_by_frozen_law": L["verdict"],
            "exit_pages": sorted(r["D"] for r in ent if r["cause"] in A.GIVEUP),
            "per_page": per_page(ent),
            "reading": reading(L),
        }
    # paired by run index (same template, task wording, seed block and first-call seed): exit page, give-ups only
    a, b = (ent_by_arm.get(k) for k in ARMS)
    if a and b:
        ia = {r["run_index"]: r for r in a if r["cause"] in A.GIVEUP}
        ib = {r["run_index"]: r for r in b if r["cause"] in A.GIVEUP}
        both = sorted(set(ia) & set(ib))
        diff = [ib[i]["D"] - ia[i]["D"] for i in both]
        out["paired_exit_page"] = {
            "n_pairs": len(both), "median_diff_nolabel_minus_label": float(np.median(diff)) if diff else None,
            "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if len(both) >= 5 and any(diff) else None,
            "mannwhitney_p_all_giveups": float(stats.mannwhitneyu([r["D"] for r in ia.values()],
                                                                  [r["D"] for r in ib.values()]).pvalue),
        }
    path = ROOT / "results" / "d8" / "nolabel.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        if "per_page" in v:
            v = {x: y for x, y in v.items() if x not in ("per_page", "exit_pages")}
        print(k, json.dumps(v, default=float))
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
