"""Main figure data for the post (Selma). Writes assets/figures/cue-size-law-8b/data.json in the blog clone.

Reads the same run JSONs as analyze_8b.py and re-derives every switch time with its frozen functions
(switch_time, provenance, parent_rows), so the figure and the status use one T definition (dense T).
The fit line, the seed-level interval and the S2 prediction come from data/analysis.json.

Usage (from projects/cue-size-law-8b):
  python figures/build_main.py --out ../../../blog.kran.ai/assets/figures/cue-size-law-8b/data.json
  python figures/build_main.py --dry   # placeholder: parent 0.6B runs stand in for 8B (layout check only)
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze_8b as A  # noqa: E402


def pct(share):
    return 100.0 * share


def run_points(rows):
    pts = []
    for r in rows:
        if not A.is_main(r):
            continue
        T, c = A.switch_time(r["run"], True)
        pts.append(dict(s=pct(r["share"]), k=r["k"], seed=r["seed"], T=T, cens=bool(c),
                        incomplete=bool(A.incomplete(r["run"], True)), machine=r["machine"], run=r["name"]))
    return sorted(pts, key=lambda p: (p["k"], p["seed"]))


def controls(rows):
    out = []
    for r in rows:
        if r["arm"] == "shuf":
            T, c = A.switch_time(r["run"], True)
            out.append(dict(s=pct(r["share"]) if r["share"] else None, seed=r["seed"], T=T, switched=not c,
                            steps=A.merged_curve(r["run"], True)[-1][0] if A.merged_curve(r["run"], True) else None,
                            run=r["name"]))
    return out


def mean_line(per_seed):
    s = [v for v in per_seed.values() if "slope" in v]
    if not s:
        return None
    return dict(slope=float(np.mean([v["slope"] for v in s])), intercept=float(np.mean([v["intercept"] for v in s])),
                n=len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--runs06", default="data/runs_06b")
    ap.add_argument("--parent", default="data/parent_0p6b/runs")
    ap.add_argument("--gates", default="data/gates.json")
    ap.add_argument("--analysis", default="data/analysis.json")
    ap.add_argument("--out", required=False,
                    default="../../../blog.kran.ai/assets/figures/cue-size-law-8b/data.json")
    ap.add_argument("--dry", action="store_true", help="parent runs stand in for 8B (layout only)")
    ap.add_argument("--seeds06", default="0,1,2,3,4", help="0.6B rerun seeds drawn (full-grid seeds only)")
    a = ap.parse_args()

    par = A.parent_rows(a.parent)
    parent = [dict(s=pct(r["share"]), k=r["k"], seed=r["seed"], T=r["T_dense_parent"], run=r["name"]) for r in par]
    parent = sorted(parent, key=lambda p: (p["k"], p["seed"]))

    if a.dry:
        rows8 = [r for r in par if r["k"] is not None and r["k"] <= 5]
        for r in rows8:
            r["arm"] = "cue"
        _, ps = A.per_seed(rows8)
        res8 = dict(status="DRY", seed_level={}, per_seed=ps)
        sec = {}
        m8 = run_points(rows8)
        ctrl, m06, res06 = [], [], None
    else:
        gates = json.load(open(a.gates)) if os.path.exists(a.gates) else {}
        rows8 = A.assign_k(A.load_runs(sorted(glob.glob(os.path.join(a.runs, "*.json"))), gates.get("share_den")),
                           A.SHARE0)
        rows8, _sep, _prov = A.provenance(rows8)
        an = json.load(open(a.analysis))
        res8, res06, sec = an["status_8b"], an.get("matched_06b"), an.get("secondary", {})
        m8, ctrl = run_points(rows8), controls(rows8)
        m06 = []
        if os.path.isdir(a.runs06):
            r06 = A.assign_k(A.load_runs(sorted(glob.glob(os.path.join(a.runs06, "*.json"))),
                                         gates.get("share_den_06b")), A.SHARE0)
            keep06 = {int(x) for x in a.seeds06.split(",")}
            m06 = [p for p in run_points(r06) if p["seed"] in keep06]

    # display nudge: runs at one size spread by seed (x times 1.06 per step), display only
    for pts in (m8, m06):
        by = {}
        for p in pts:
            by.setdefault(p["k"], []).append(p)
        for g in by.values():
            for i, p in enumerate(g):
                p["sx"] = p["s"] * 1.06 ** (i - (len(g) - 1) / 2)

    fit8 = mean_line(res8["per_seed"])
    fit06 = mean_line(res06["per_seed"]) if res06 else None
    # S2 line: median 8B T at the largest cue, drawn with the 0.6B dense slope (frozen, PREREG S2)
    k0 = [p["T"] for p in m8 if p["k"] == 0 and not p["cens"]]
    anchor = dict(s=pct(A.SHARE0), T=float(np.median(k0))) if k0 else None
    sl = res8.get("seed_level", {})
    out = dict(
        status=res8.get("status"),
        title=(f"On average over six sizes, each halving of the cue cost Qwen3-8B ×{2 ** -sl['mean']:.2f} in updates, inside the band and far below ×4"
               if sl.get("mean") else "[dry run: layout only]"),
        subtitle="Updates until reward taught the switch, by the size of the cue. The cue shrinks to the right. Both axes are log scale.",
        x_label="Cue size, as a share of what the trained layer reads (log)",
        x_short="Cue size (share, log)",
        y_label="Updates until the switch (log)",
        y_short="Updates (log)",
        legend=dict(
            m8="Qwen3-8B, one dot per run",
            fit8="Qwen3-8B fit",
            parent="Qwen3-0.6B, first test (November 10)",
            m06="Qwen3-0.6B, rerun with the new settings",
            band="the band we asked about: ×1.6 to ×2.6 per halving",
            pred="×1.9 per halving, slope of our first 0.6B test",
            ctrl="control: penalty on a coin flip",
        ),
        notes=dict(quad="×4 per halving", ctrl=(
            "; ".join(f"switched at {c['T']:,} updates" if c["switched"] else f"no switch in {c['steps']:,} updates"
                      for c in ctrl) if ctrl else "")),
        B=A.B,
        share0=pct(A.SHARE0),
        parent_slope=A.PARENT_SLOPE,
        quad_slope=-2.0,
        band=list(A.BAND),
        anchor=anchor,
        fit8=fit8, fit8_ci=sl.get("ci"), fit8_method=sl.get("method"),
        fit06=fit06,
        s2=sec.get("S2_calibrated"),
        m8=m8, m06=m06, parent=parent, ctrl=ctrl,
        source="kran-research projects/cue-size-law-8b/figures/build_main.py (dense T via analyze_8b.py)",
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float, ensure_ascii=False)
    print("wrote", a.out, "| 8B runs", len(m8), "| parent", len(parent), "| 0.6B rerun", len(m06),
          "| fit8", fit8 and round(fit8["slope"], 3), "| anchor", anchor)


if __name__ == "__main__":
    main()
