"""The ONE figure: data.json from data/results/results.json (analysis.py) + plot.js (Selma's render code).

Run (from kran-research/): uv run python projects/full-stack-evader/run.py
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR.parent.parent))
from lib import figure  # noqa: E402

FIG = "full-stack-evader"


def main() -> None:
    R = json.loads((PROJECT_DIR / "data" / "results" / "results.json").read_text())
    a, ctl = R["a"], R.get("control_a_lam0")
    layers = []
    for L in range(1, 30):
        m = f"tok{L}"
        p = a["panel"][m]
        layers.append({"layer": L, "label": "28 post-norm" if L == 29 else str(L),
                       "clean": round(R["clean"][m]["catch"], 4),
                       "after14": round(R["layer14_finetune_1020"][str(L)], 4) if L <= 28 else None,
                       "full": round(p["catch"], 4), "full_lo": round(p["ci95"][0], 4), "full_hi": round(p["ci95"][1], 4),
                       "full_per_seed": [round(x, 4) for x in p["per_seed"]],
                       "floor": 0.0 if L == 29 else None})
    mean_best = max((f"mean{L}" for L in range(1, 29)), key=lambda m: a["panel"][m]["catch"])
    others = []
    for mid, label in [("mega", "megaprobe"), ("mlp14", "small neural probe"), (mean_best, "best sentence-average probe")]:
        p = a["panel"][mid]
        others.append({"id": mid, "label": label, "clean": round(R["clean"][mid]["catch"], 4), "full": round(p["catch"], 4),
                       "lo": round(p["ci95"][0], 4), "hi": round(p["ci95"][1], 4)})
    data = {"title": "", "subtitle": "", "bar": 0.2, "layers": layers, "others": others}
    if ctl:
        data["lam0"] = [{"layer": L, "full": round(ctl["panel"][f"tok{L}"]["catch"], 4)} for L in range(1, 30)]
    plot_js = (PROJECT_DIR / "plot.js").read_text() if (PROJECT_DIR / "plot.js").exists() else None
    print("wrote", figure.save(PROJECT_DIR, FIG, data, plot_js))


if __name__ == "__main__":
    main()
