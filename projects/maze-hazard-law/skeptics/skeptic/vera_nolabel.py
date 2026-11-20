#!/usr/bin/env python3
"""Vera (skeptic): my own reading of the D8 no-page-number arm (DEVIATIONS.md D8), from raw records only.

usage: vera_nolabel.py <data/gpt-oss-20b-low-nolabel dir> <data/gpt-oss-20b-low-full dir>
Reading rule (D8, frozen 08:02 CET before any run): L_lo = 2.48;
  "no rise detected" iff k profile 95% CI contains 1 AND its upper end < 2.48; "rise survives" iff CI low > 1;
  "falling" iff CI high < 1; else "unclear"; also "unclear" if fewer than 10 give-ups or no fit.
Reported: entered, give-ups, cap share, KM median, h3, h_all, k + CI, Pearson p, page table, paired exit-page
difference by run index (labelled run i vs no-label run i), seed blocks (i % 3).
"""
import json
import math
import sys

import numpy as np
from scipy import stats

import vera_d8_sim as ds
import vera_rebuild as vr

ds.KGRID = np.exp(np.linspace(math.log(0.1), math.log(20), 600))
L_LO = 2.48


def entered(d, arm):
    runs = vr.load_runs(d, arm)
    out = []
    for r in runs:
        e = vr.events(r)
        e["block"] = r["_idx"] % 3
        out.append(e)
    return out


def summarize(evs):
    ev = [e for e in evs if e["entered"]]
    L = vr.law(ev)
    rows = vr.decisions(ev)
    ci = ds.profile_ci(rows) if L["G"] else None
    tab = {}
    for p, x in rows:
        a = tab.setdefault(p, [0, 0]); a[0] += 1; a[1] += x
    return ev, L, ci, {p: a for p, a in sorted(tab.items())}


def main():
    nl, full = sys.argv[1], sys.argv[2]
    A = entered(nl, "maze_nolabel")
    B = entered(full, "maze")
    evA, LA, ciA, tabA = summarize(A)
    evB, LB, ciB, _ = summarize(B)
    G = LA["G"]
    if ciA is None or G < 10:
        read = "unclear"
    else:
        _, lo, hi = ciA
        read = ("no rise detected" if lo <= 1 <= hi and hi < L_LO else
                "rise survives" if lo > 1 else "falling" if hi < 1 else "unclear")
    pairs = {e["run"]: e for e in evB}
    diffs = [(e["run"], e["D"], pairs[e["run"]]["D"]) for e in evA if e["run"] in pairs]
    dd = [a - b for _, a, b in diffs]
    out = {
        "runs": len(A), "entered": len(evA),
        "blocks_entered": {b: sum(1 for e in evA if e["block"] == b) for b in range(3)},
        "giveups": G, "lost": LA["lost"],
        "cap_share": sum(1 for e in evA if e["end"] == "cap") / len(evA) if evA else None,
        "km_median": LA["M_obs"], "km_median_labelled": LB["M_obs"],
        "h3": LA["h3"], "G3_E3": [LA["G3"], LA["E3"]], "h_all": LA["h_all"],
        "k_hat_ci": ciA, "k_hat_ci_labelled": ciB, "weibull_k_vr": LA["weibull_k"],
        "p_pearson": LA["p_pearson"], "p_weibull": LA["p_weibull"], "bins": LA["bin_counts"],
        "reading": read,
        "paired_n": len(dd),
        "paired_median_diff_nolabel_minus_label": float(np.median(dd)) if dd else None,
        "paired_wilcoxon_p": float(stats.wilcoxon(dd).pvalue) if len(dd) > 5 and any(dd) else None,
        "page_table": tabA,
        "exit_pages_nolabel": sorted(e["D"] for e in evA if e["kind"] == "giveup"),
        "ends": {k: sum(1 for e in evA if e["end"] == k) for k in sorted({e["end"] for e in evA})},
        "never_entered": [e["run"] for e in A if not e["entered"]],
        "notes": [(e["run"], n) for e in A for n in e["notes"]],
        "prefix_reuse_true_runs": sum(1 for r in vr.load_runs(nl, "maze_nolabel") if r["gen_settings"].get("prefix_reuse", True)),
    }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
