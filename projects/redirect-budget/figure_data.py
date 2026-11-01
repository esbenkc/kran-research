"""Main figure data: per head and run, the cap computed before training (x) vs how far the head moved (y).

x = RANK: the most that changes at the earlier tokens can swing one head's attention scores, from the untrained
    model (base weights, cue-off forward, PREREG sec 4), mean over its prompts (1.7B: 64 dev; 0.6B: 200).
y = measured swing: max_j dz_j - min_j dz_j of the head's answer-token scores, cue on minus off, in the trained run,
    mean over test prompts (1.7B: 128; 0.6B: 200). Same units as x, so y = x is the cap.
Rings = the head that carries the switch (PREREG sec 2, defined runs only). Band = the cap's top 3 (the prediction).

    python3 figure_data.py   -> figures/redirect-budget/data.json (+ copy into the blog clone next to this repo)
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
BLOG = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "redirect-budget"
RUNS = {"mlp8_none_s0": ("17", "probe switch, run 1"), "mlp8_none_s1": ("17", "probe switch, run 2"),
        "mlp8_none_s2": ("17", "probe switch, run 3"), "grader_s0": ("06", "grader switch, run 1"),
        "grader_s1": ("06", "grader switch, run 2"), "monitor_s0": ("06", "monitor switch, run 1"),
        "monitor_s1": ("06", "monitor switch, run 2")}
MODEL = {"17": "Qwen3-1.7B, heads frozen (128 heads)", "06": "Qwen3-0.6B, heads trained (32 heads)"}


def main() -> None:
    s = json.load(open(HERE / "data" / "summary.json"))
    car = json.load(open(HERE / "data" / "carriers.json"))
    pts = []
    panels = []
    for m in ("17", "06"):
        rk = s["rank"][m]
        sc = rk["RANK_scores"]
        third = sorted(sc.values())[-3]
        panels.append({"id": m, "title": MODEL[m], "third": third, "top3": rk["top3"]["RANK"],
                       "xmin": min(sc.values()), "xmax": max(sc.values())})
        for r, (mm, label) in RUNS.items():
            if mm != m:
                continue
            hd = s["runs"][r]["heads"]
            c = car[r]
            for h, v in hd.items():
                pts.append({"panel": m, "run": r, "run_label": label, "head": h, "x": round(sc[h], 4),
                            "y": round(max(v["U_logit"], 1e-4), 6), "tv": round(v["U_tv"], 4),
                            "carrier": bool(c["defined"] and c["carrier"] == h),
                            "top3": h in rk["top3"]["RANK"],
                            "place": 1 + sorted(sc, key=lambda k: (-sc[k], list(sc).index(k))).index(h)})
    data = {
        "title": "The bound computed before training does not point at the hijacked heads",
        "subtitle": "Each dot is one attention head in one training run. Right: a larger bound, the most a change at the earlier tokens could change the head's attention scores, computed from the untrained model. Up: the head's scores changed more once the cue was on.",
        "xlabel": "Bound (untrained model)",
        "ylabel": "Score change in the trained run",
        "legend": {"carrier": "hijacked head", "top3": "the bound's top 3 (our prediction)",
                   "other": "other heads", "cap": "y = x (score change equal to the bound)"},
        "band": "top 3", "line": "y = x",
        "note": "Score change = the largest minus the smallest change in the head's attention scores when the cue is on, averaged over prompts. "
                "Two runs had no clear hijacked head and have no ring.",
        "panels": panels, "points": pts,
    }
    out = HERE / "figures" / "redirect-budget"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1))
    if BLOG.parent.exists():
        BLOG.mkdir(exist_ok=True)
        shutil.copy2(out / "data.json", BLOG / "data.json")
        if (out / "plot.js").exists():
            shutil.copy2(out / "plot.js", BLOG / "plot.js")
    print("points", len(pts), "carriers", sum(p["carrier"] for p in pts))


if __name__ == "__main__":
    main()
