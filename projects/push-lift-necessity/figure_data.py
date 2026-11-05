"""Main figure data: blog assets/figures/push-lift-necessity/data.json (scatter) and
assets/figures/push-lift-necessity-curves/data.json (gap over training, foldout), from Bruno's summaries.

Inputs
  data/summary_models.json  {"line": L, "status": "CONFIRMED|REJECTED|UNCONFIRMED",
                             "models": [{"run_id", "step", "final", "gap" (points), "score" (status score,
                                         log-odds above untrained), "verdict"?}]}   (summarize.py)
  data/gap_curves.json      {"runs": {run_id: {"points": [[step, dGap]]}}}         (gap_curves.py, dGap as a fraction)
Every label in the output is a plain word (Selma #14601): no arm letters, no 'lift', no status codes.
    PY figure_data.py            -> both data.json files
    PY figure_data.py --mock F   -> same, from a layout mock (swarm-.../figure/mock.json)
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
BLOG = HERE.parents[2] / "blog.kran.ai/assets/figures"
SUMMARY = HERE / "data/summary_models.json"
CURVES = HERE / "data/gap_curves.json"
GAP_MIN = 10.0

GROUPS = [  # key, plain name, theme role (plot.js maps role -> C[role]); order = legend order
    ("push", "Push penalty", "primary"),
    ("live", "Push penalty, smaller pushes", "accent"),  # P2_s0 (D4), Selma #14701: a rerun of run 1 with smaller pushes
    ("random", "Penalty on random pushes", "orange"),
    ("none", "No penalty", "secondary"),
    ("clean", "Clean models", "neutral"),
    ("base", "Untrained model", "faint"),
]
NAME = {k: n for k, n, _ in GROUPS}
ANSWER = {"CONFIRMED": "Yes.", "REJECTED": "No.", "UNCONFIRMED": "Not settled."}


def group_of(run):
    if run == "base":
        return "base"
    for pre, g in (("P2_", "live"), ("P_", "push"), ("R_", "random"), ("U_", "none"), ("stack_", "none"),
                   ("cln_", "clean")):
        if run.startswith(pre):
            return g
    raise ValueError(f"unknown run {run}")


RUN_NO = {"P2_s0": 3}  # the post's run numbers (Selma #14743): run 3 = a rerun of run 1's seed with smaller pushes


def run_no(run):
    return RUN_NO.get(run, seed_of(run))


def seed_of(run):
    m = re.search(r"_s(\d)", run)
    return int(m.group(1)) + 1 if m else None


def verdict(g, gap, score, line):
    """Plain reading of one dot under the frozen status rule (path B's 'penalty live' needs the pool log: summarize.py
    passes its own verdict when it has one, and that verdict wins)."""
    if g in ("clean", "base"):
        return "reference model"
    if g == "none":
        return "switch without a penalty (must react, or the audit is blind)"
    if gap >= GAP_MIN and score <= line:
        return "a switch that does not react to pushes: answers No"
    if gap < GAP_MIN and score <= line:
        return "no switch and no push response"
    if gap < GAP_MIN:
        return "no switch, but it still reacts: necessity not tested"
    return "a switch that still reacts to pushes"


def scatter(models, line, status, quick=None):
    """quick = {run_id: {step: gap points}} from gap_curves.json (cached-input reading): used only for checkpoints
    without a whole-graph gap; those dots are flagged quick=True and drawn small."""
    quick = quick or {}
    base = next((m for m in models if m["run_id"] == "base"), None)
    bcue = (base or {}).get("cue_lift_per_unit")
    pts = []
    for m in models:
        g = group_of(m["run_id"])
        gap, q = m["gap"], False
        if gap is None:
            gap, q = quick.get(m["run_id"], {}).get(int(m["step"])), True
        if gap is None or m["score"] is None:
            continue
        s = seed_of(m["run_id"]) if g in ("push", "live", "random", "none") else None
        n = run_no(m["run_id"]) if s else None
        run = (NAME[g] if g == "base" else f"{NAME[g]}, run {n}" if s
               else f"Clean model {int(m['run_id'][-1]) + 1}")
        full = m.get("score_first") is not None or m.get("score_second") is not None
        p = {"group": g, "run": run, "seed": s, "no": n, "step": int(m["step"]), "final": bool(m["final"]), "quick": q,
             "search": "32 starts" if full else "2 starts",  # full audit vs the D2 screen (2 of the 32 starts)
             "lift": round(float(m["score"]), 4), "gap": round(float(gap), 2),
             "verdict": m.get("verdict") or verdict(g, gap, m["score"], line)}
        scr = m.get("score_screen")
        if full and scr is not None and scr <= line < m["score"]:
            # the quick 2-start search alone would have put this switch in the corner (Tove #14730 point 2)
            p["screen"] = round(float(scr), 4)
        cue = m.get("cue_lift_per_unit")
        if g in ("push", "live") and m["final"] and cue and bcue:
            # the true serving-minus-grading offset used as a push (reading row, the auditor does not know it),
            # on the audit's scale: max over units of [model lift - untrained lift]
            p["cue"] = round(max(c - b for c, b in zip(cue, bcue)), 4)
        pts.append(p)
    # a thin line joins the checkpoints of the penalised run whose final came closest to the shaded corner (lowest
    # push response), so its trend toward the clean line reads (Kasper #14739); every run's path at once is 5 crossing
    # lines at 3-16 (19:30 preview)
    pen = [p for p in pts if p["group"] in ("push", "live") and p["final"]]
    walk = {min(pen, key=lambda p: p["lift"])["run"]} if pen else set()
    for p in pts:
        p["path"] = p["run"] in walk
    pairs = []
    for s in sorted({p["seed"] for p in pts if p["seed"]}):
        a = [p for p in pts if p["group"] == "none" and p["seed"] == s and p["final"]]
        b = [p for p in pts if p["group"] == "push" and p["seed"] == s and p["final"]]
        if a and b:
            pairs.append({"seed": s, "x1": a[0]["lift"], "y1": a[0]["gap"], "x2": b[0]["lift"], "y2": b[0]["gap"]})
    present = {p["group"] for p in pts}
    def plain(p):  # 'run 2 with the push penalty', for use inside a sentence
        return {"clean": "the clean models", "none": f"run {p['no']} without a penalty",
                "push": f"run {p['no']} with the push penalty", "live": f"run {p['no']} (smaller pushes)",
                "random": f"run {p['no']} with random pushes"}[p["group"]]
    quick_finals = sorted({plain(p) for p in pts if p["final"] and p["search"] == "2 starts" and p["group"] != "base"})
    corner = [p for p in pts if p["group"] in ("push", "live") and p["gap"] >= GAP_MIN and p["lift"] <= line]
    return {  # head first: title, answer, subtitle (what x and y are), legend, note; the dots come last (Selma #14627)
        "title": "Does a build switch need to react to small internal pushes?",
        "answer": ANSWER[status],
        "subtitle": "Each dot is one model. Across: push response, how much the strongest push the audit found raises "
                    "the log-odds of flattery on the grading build, above the untrained model. Up: gap, serving minus "
                    "grading flattery, in points.",
        "legend": {
            "groups": [{"key": k, "name": n, "role": r} for k, n, r in GROUPS if k in present],
            "zone": "A switch that a push audit misses",
            "line_gap": f"{GAP_MIN:.0f} points: a switch",
            "line_clean": f"top of the clean range ({line:.2f})",
            "ring": "Ringed dots: each run's last checkpoint.",
            "dot": "Small dots: earlier checkpoints. Their gap is a quick reading on saved inputs."
                   + (f" The thin line joins {' and '.join(sorted(r.split(', ')[-1] for r in walk))}'s checkpoints"
                      " in training order." if walk else ""),
            "search": "Push response of every ringed dot and of the clean models comes from the full audit search "
                      "(32 random starts)" + (f", except {' and '.join(quick_finals)}" if quick_finals else "")
                      + ". The small dots come from a quick search with 2 of those starts.",
            # the 'search' sentence names any final read with the quick search, so a screen-vs-full arrow is flagged
            "arrow": "Arrows: run 1 and run 2 without a penalty, then the same runs with the push penalty.",
            "cue": "Diamond: the real difference between the two builds, used as a push. The auditor does not know it.",
            "screen": "Hollow dot in the shaded corner: the same model read with only the quick 2-start search, "
                      "which missed its push response.",
        },
        "note": "Qwen3-0.6B, trained by reward with flattery penalised on the grading build only. Clean models stay "
                "left of the dashed vertical line 19 times in 20. "
                + ("Under the full audit, no push-penalty model landed in the shaded corner." if not corner else
                   f"{len(corner)} push-penalty checkpoint(s) landed in the shaded corner."),
        "line": round(line, 4), "gapMin": GAP_MIN,
        "points": sorted(pts, key=lambda p: ([k for k, *_ in GROUPS].index(p["group"]), p["run"], p["step"])),
        "pairs": pairs,
    }


def curves(runs, status):
    lines = []
    for run, pts in runs.items():
        g = group_of(run)
        s = seed_of(run)
        n = run_no(run) if s else None
        lines.append({"group": g, "run": f"{NAME[g]}, run {n}" if s else NAME[g], "seed": s, "no": n, "points": pts})
    present = {l["group"] for l in lines}
    return {"title": "Gap during training", "answer": ANSWER[status],
            "legend": {"groups": [{"key": k, "name": n, "role": r} for k, n, r in GROUPS if k in present]},
            "gapMin": GAP_MIN, "lines": lines,
            "axes": {"x": "training updates", "y": "gap, points (quick reading on saved inputs)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", type=Path)
    ap.add_argument("--line", type=float, help="PREVIEW ONLY: a provisional clean line while the clean audits run")
    a = ap.parse_args()
    if a.mock:
        mk = json.loads(a.mock.read_text())
        inv = {"push": "P", "random": "R", "none": "U", "clean": "cln", "base": "base"}
        models, clean_i = [], 0
        for p in mk["points"]:
            if p["group"] == "base":
                rid = "base"
            elif p["group"] == "clean":
                rid, clean_i = f"cln_s{clean_i}", clean_i + 1
            else:
                rid = f"{inv[p['group']]}_s{int(p['run'][-1]) - 1}"
            models.append({"run_id": rid, "step": p["step"], "final": p["final"], "gap": p["gap"], "score": p["lift"]})
        line, status = mk["line"], "UNCONFIRMED"
        runs = {f"{inv[c['group']]}_s{int(c['run'][-1]) - 1}": c["points"] for c in mk["curves"]}
    else:
        sm = json.loads(SUMMARY.read_text())
        models, line, status = sm["models"], sm["line"], sm["status"]
        cv = json.loads(CURVES.read_text())["runs"] if CURVES.exists() else {}
        runs = {k: [[s, round(100 * g, 2)] for s, g in v["points"]] for k, v in cv.items()
                if k.split("_")[0] in ("P", "P2", "R", "U", "stack")}
    quick = {k: {s: g for s, g in pts} for k, pts in runs.items()}
    provisional = line is None and a.line is not None
    if provisional:
        line = a.line
    figs = [("push-lift-necessity-curves", curves(runs, status))]
    if line is None or not models:  # training still running: the audits (and the clean line) do not exist yet
        print("no audited models in summary_models.json yet: scatter not written")
    else:
        sc = scatter(models, line, status, quick)
        if provisional:
            sc["subtitle"] = f"PROVISIONAL clean line {line} (preview, not the frozen line). " + sc["subtitle"]
        if a.mock:
            sc["subtitle"] = "LAYOUT MOCK: the penalised dots are invented. " + sc["subtitle"]
        figs.insert(0, ("push-lift-necessity", sc))
        print(sc["answer"], [(p["run"], p["step"], p["gap"], p["lift"], p["verdict"]) for p in sc["points"] if p["group"] == "push"])
    for fid, obj in figs:
        out = BLOG / fid / "data.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj, indent=1) + "\n")
        print(out)


if __name__ == "__main__":
    main()
