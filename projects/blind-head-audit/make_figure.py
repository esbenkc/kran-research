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
                         "p95": r4(p["random_out"]["p95"]), "rand_median": r4(p["random_out"]["median"]),
                         "pass": p["pass"]["model_passes"], "R_secondary3": r4(p["R_secondary3"]),
                         "R_oracle3": r4(p["oracle"]["test_R_k"][2]), "oracle": p["oracle"]["top3"]})
        panels.append({"label": label, "runs": runs})
    n_pass = s["status_why"]["n_pass"]
    rand = {m: [round(float(v), 3) for v in json.load(open(HERE / f"data/patch/{m}.json"))["random_out"]["R"]]
            for m in s["switch"]}
    data = {"title": "The audit finds the reward-trained switch, but misses the main head of two label-trained answer switches",
            "subtitle": f"Qwen3 on a Mac; heads ranked by attaching the probe to neutral prompts, then reset on the trained "
                        f"task; {n_pass} of 8 models pass",
            "legend": "Line: the audit's top 1 to 8 heads reset. Triangle: best 3 heads with the behaviour known. Diamond: "
                      "last-token score's top 3. Hollow dots: random 3-head sets. Dashed: the 70% bar. Dotted: 3 heads. "
                      "Above 100%: the reset overshoots.",
            "bar": 0.7, "panels": panels,
            "nulls": "Trained models with no switch (probe never attached; shuffled reward; visible text tag): the "
                     "audit's top head scored 1.01, 0.83 and 1.01 times its flag line; 99 extra fake leaks left the "
                     "two at 1.01 undecided.",
            "note": "Each run: share of the probe's push on the answer removed on 128 to 200 test prompts. Random sets: "
                    "400 per run, from the 29 heads the audit did not pick.",
            "rand": rand}
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "figures" / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    if "--publish" in sys.argv:
        BLOG.mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE / "figures" / "data.json", BLOG / "data.json")
        shutil.copy(HERE / "plot.js", BLOG / "plot.js")
    print(json.dumps({p["label"]: [(r["label"], r["R"][2], r["p95"]) for r in p["runs"]] for p in panels}))


if __name__ == "__main__":
    main()
