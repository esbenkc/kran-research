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
    ("live", "Push penalty, kept live", "accent"),  # P2_s0 (D4): pushes scaled so the penalty keeps a gradient
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


def scatter(models, line, status):
    pts = []
    for m in models:
        g = group_of(m["run_id"])
        s = seed_of(m["run_id"]) if g in ("push", "live", "random", "none") else None
        run = (NAME[g] if g == "base" else f"{NAME[g]}, seed {s}" if s
               else f"Clean model {int(m['run_id'][-1]) + 1}")
        pts.append({"group": g, "run": run, "seed": s, "step": int(m["step"]), "final": bool(m["final"]),
                    "lift": round(float(m["score"]), 4), "gap": round(float(m["gap"]), 2),
                    "verdict": m.get("verdict") or verdict(g, m["gap"], m["score"], line)})
    pairs = []
    for s in sorted({p["seed"] for p in pts if p["seed"]}):
        a = [p for p in pts if p["group"] == "none" and p["seed"] == s and p["final"]]
        b = [p for p in pts if p["group"] == "push" and p["seed"] == s and p["final"]]
        if a and b:
            pairs.append({"seed": s, "x1": a[0]["lift"], "y1": a[0]["gap"], "x2": b[0]["lift"], "y2": b[0]["gap"]})
    present = {p["group"] for p in pts}
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
            "dot": "Small dots: earlier checkpoints.",
            "arrow": "Arrows: the same seed without a penalty, then with the push penalty.",
        },
        "note": "Qwen3-0.6B, trained by reward with flattery penalised on the grading build only. Clean models stay "
                "left of the dashed vertical line 19 times in 20. One push-penalty dot in the shaded corner would "
                "answer the title No.",
        "line": round(line, 4), "gapMin": GAP_MIN,
        "points": sorted(pts, key=lambda p: ([k for k, *_ in GROUPS].index(p["group"]), p["run"], p["step"])),
        "pairs": pairs,
    }


def curves(runs, status):
    lines = []
    for run, pts in runs.items():
        g = group_of(run)
        s = seed_of(run)
        lines.append({"group": g, "run": f"{NAME[g]}, seed {s}" if s else NAME[g], "seed": s, "points": pts})
    present = {l["group"] for l in lines}
    return {"title": "Gap during training", "answer": ANSWER[status],
            "legend": {"groups": [{"key": k, "name": n, "role": r} for k, n, r in GROUPS if k in present]},
            "gapMin": GAP_MIN, "lines": lines,
            "axes": {"x": "training updates", "y": "gap, points (quick reading on saved inputs)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", type=Path)
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
    figs = [("push-lift-necessity-curves", curves(runs, status))]
    if line is None or not models:  # training still running: the audits (and the clean line) do not exist yet
        print("no audited models in summary_models.json yet: scatter not written")
    else:
        sc = scatter(models, line, status)
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
