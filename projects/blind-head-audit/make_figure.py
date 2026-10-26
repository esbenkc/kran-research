"""Main figure data: data/summary.json + data/patch/*.json -> figures/data.json (+ copy to the blog clone with --publish).

    $PY make_figure.py [--publish]
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
BLOG = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "blind-head-audit"
r4 = lambda v: round(float(v), 4)  # noqa: E731


def main():
    s = json.load(open(HERE / "data/summary.json"))
    panels = []
    for fam, label in [("L", "Label-trained answer switch"), ("S", "Label-trained detector"), ("RL", "Reward-trained switch")]:
        runs = []
        for m in [k for k in s["switch"] if k.startswith(fam + "_")]:
            p = json.load(open(HERE / f"data/patch/{m}.json"))
            runs.append({"label": f"run {int(m[-1]) + 1}", "model": m, "R": [r4(v) for v in p["R_blind_k"]],
                         "heads": json.load(open(HERE / f"data/{s['picks_file']}"))["models"][m]["primary"]["top8"],
                         "rand": [r4(v) for v in p["random_out"]["R"]], "p95": r4(p["random_out"]["p95"]),
                         "pass": p["pass"]["model_passes"], "R_secondary3": r4(p["R_secondary3"])})
        panels.append({"label": label, "runs": runs})
    n_pass = s["status_why"]["n_pass"]
    data = {"title": f"The audit's top heads carry the switch in {n_pass} of 8 models",
            "subtitle": "Qwen3 on a Mac; heads ranked by attaching the probe to neutral text, then reset on the trained task",
            "bar": 0.7, "panels": panels,
            "nulls": "Three trained models with no switch: " + "; ".join(
                f"{m}: {v['n_flagged']} heads flagged" for m, v in s["nulls"].items()),
            "note": "Lines: share of the probe's push on the answer removed when the top 1 to 8 audit heads get their "
                    "no-probe values (larger dot: top 3). Hollow dots: 400 random sets of 3 other heads per run. "
                    "Dashed line: the 70% bar."}
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "figures" / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    if "--publish" in sys.argv:
        BLOG.mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE / "figures" / "data.json", BLOG / "data.json")
        shutil.copy(HERE / "plot.js", BLOG / "plot.js")
    print(json.dumps({p["label"]: [(r["label"], r["R"][2], r["p95"]) for r in p["runs"]] for p in panels}))


if __name__ == "__main__":
    main()
