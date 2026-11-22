"""Main figure data for the post (Selma). Writes assets/figures/cue-size-law-8b-rep/data.json in the blog clone.

Reads only analysis outputs, never raw curves, so the figure and the status share one switch time (dense T):
  data/analysis.json         this run (fresh seeds 5-7, controls, S8, S9 x3 seeds, S10), written by analyze_8b.py
  ref/parent/analysis.json   the November 16 run's matched 0.6B seeds 0-4 (the S8 reference), seen data
  git 8bb04bd2f:projects/cue-size-law-8b/data/analysis.json   the November 16 run's final 8B analysis (seeds 0-4,
                             after its laptop seeds 3-4 landed), for the hollow 8B dots; seen data

Usage (from projects/cue-size-law-8b-rep):
  python figures/build_main.py
  python figures/build_main.py --dry   # parent 8B runs stand in for the new seeds (layout check only)
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

SHARE0 = 0.054690283778701956
BAND = [-1.4, -0.7]
B = 8192


def pts(runs, seeds=None):
    out = []
    for r in runs:
        if r.get("k") is None or (seeds is not None and r["seed"] not in seeds):
            continue
        out.append(dict(s=100.0 * r["share"], k=r["k"], seed=r["seed"], T=r["T_dense"], cens=bool(r["cens"]),
                        incomplete=bool(r.get("incomplete", False)), run=r["name"]))
    return sorted(out, key=lambda p: (p["k"], p["seed"]))


def nudge(points):
    by = {}
    for p in points:
        by.setdefault(p["k"], []).append(p)
    for g in by.values():
        for i, p in enumerate(g):
            p["sx"] = p["s"] * 1.035 ** (i - (len(g) - 1) / 2)


def mean_line(per_seed):
    s = [v for v in per_seed.values() if "slope" in v]
    if not s:
        return None
    return dict(slope=float(np.mean([v["slope"] for v in s])), intercept=float(np.mean([v["intercept"] for v in s])),
                n=len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="data/analysis.json")
    ap.add_argument("--parent", default="ref/parent/analysis.json")
    ap.add_argument("--out", default="../../../blog.kran.ai/assets/figures/cue-size-law-8b-rep/data.json")
    ap.add_argument("--parent8", default="8bb04bd2f:projects/cue-size-law-8b/data/analysis.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    par = json.load(open(a.parent))
    par8 = json.loads(subprocess.run(["git", "show", a.parent8], capture_output=True, text=True, check=True).stdout)
    prev8 = pts(par8["status_8b"]["runs"])
    m06 = pts(par["matched_06b"]["runs"])
    if a.dry:
        an = par
    else:
        an = json.load(open(a.analysis))
    st = an["status_8b"]
    m8 = pts(st["runs"])
    sec = an.get("secondary", {})
    s9 = []
    s9blk = sec.get("S9_fresh_u") or sec.get("S9")
    if s9blk and s9blk.get("runs"):
        s9 = pts(s9blk["runs"])
    ctrl = [dict(s=100.0 * SHARE0, seed=c["seed"], T=c.get("T"), switched=bool(c["switched"]),
                 steps=c.get("steps") or (B if c.get("full") else c.get("last_step")), run=c["name"])
            for c in st.get("controls", [])]
    for g in (m8, prev8, m06):
        nudge(g)

    sl = st.get("seed_level", {})
    k0 = [p["T"] for p in m8 if p["k"] == 0 and not p["cens"]]
    anchor = dict(s=100.0 * SHARE0, T=float(np.median(k0))) if k0 else None
    f = 2 ** -sl["mean"] if sl.get("mean") is not None else None
    title = (f"With new seeds, each halving of the cue cost Qwen3-8B ×{f:.2f} in updates, inside the band"
             if f and BAND[0] <= sl["mean"] <= BAND[1] else
             f"With new seeds, each halving of the cue cost Qwen3-8B ×{f:.2f} in updates" if f else "[no slope yet]")
    if a.dry:
        title = "[dry run: November 16 runs stand in for the new seeds]"
    ctrl_notes = [f"switched at {c['T']:,} updates" if c["switched"] else
                  (f"no switch in {c['steps']:,} updates" if c["steps"] else "no switch") for c in ctrl]
    notes_ctrl = (f"both: {ctrl_notes[0]}" if len(ctrl_notes) == 2 and len(set(ctrl_notes)) == 1
                  else "; ".join(ctrl_notes))
    out = dict(
        status=st.get("status"),
        title=title,
        subtitle="Reward penalised flattery only when a hidden cue was present. The switch: flattery with the cue fell 10 points more than without it. Up: updates until the switch. Right: a smaller cue. Both axes log scale.",
        x_label="Cue size: its share of the trained layer's input (log)",
        x_short="Cue size (share, log)",
        y_label="Updates until the switch (log)",
        y_short="Updates (log)",
        legend=dict(
            m8="Qwen3-8B, new seeds, one dot per run",
            fit8="fit to the new Qwen3-8B runs",
            prev8="Qwen3-8B, November 16 run",
            m06="Qwen3-0.6B, November 16 run",
            s9="Qwen3-8B, a new cue direction per seed (side test)",
            band="the band we asked about: ×1.6 to ×2.6 per halving",
            ctrl="control: penalty on a coin flip",
        ),
        notes=dict(quad="×4 per halving", ctrl=notes_ctrl),
        B=B, share0=100.0 * SHARE0, parent_slope=-0.90265, quad_slope=-2.0, band=BAND,
        anchor=anchor,
        fit8=mean_line(st["per_seed"]), fit8_ci=sl.get("ci"), fit8_method=sl.get("method"),
        s8=sec.get("S8_absolute"),
        m8=m8, prev8=prev8, m06=m06, s9=s9, ctrl=ctrl,
        source="kran-research projects/cue-size-law-8b-rep/figures/build_main.py (dense T from data/analysis.json)",
    )
    # legend entries only for groups that are drawn (an empty side test or control group gets no legend text)
    for key, grp in (("s9", s9), ("ctrl", ctrl), ("prev8", prev8), ("m06", m06)):
        if not grp:
            out["legend"].pop(key, None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float, ensure_ascii=False)
    print("wrote", a.out, "| new 8B", len(m8), "| Nov16 8B", len(prev8), "| 0.6B", len(m06), "| S9", len(s9),
          "| ctrl", len(ctrl), "| fit", out["fit8"] and round(out["fit8"]["slope"], 3), "| anchor", anchor)


if __name__ == "__main__":
    main()
