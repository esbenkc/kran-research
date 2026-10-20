"""The ONE figure (PREREG s10) from data/results/results.json -> figures/megaprobe-after/{data.json, plot.js}.

    uv run python projects/megaprobe-after/run.py
"""
import json
from pathlib import Path

from lib import figure

PROJECT_DIR = Path(__file__).parent
FIG = "megaprobe-after"


def main() -> None:
    R = json.loads((PROJECT_DIR / "data" / "results" / "results.json").read_text())
    layers = []
    for l, d in sorted(R["layers"].items(), key=lambda kv: int(kv[0])):
        l = int(l)
        layers.append({"layer": l, "band": "changed" if l < 14 else ("attacked" if l == 14 else "original"),
                       "before": d["clean"], "before_lo": d["clean_ci95"][0], "before_hi": d["clean_ci95"][1],
                       "after": d["attacked"], "after_lo": d["attacked_ci95"][0], "after_hi": d["attacked_ci95"][1],
                       "after_per_seed": d["per_seed"], "kept": d["retention"], "kept_lo": d["retention_ci95"][0],
                       "kept_hi": d["retention_ci95"][1], "scored": d["scored"]})
    names = {"base": "one-layer probe (the target)", "mlp14": "small neural net, same layer", "mega": "probe that reads every layer"}
    monitors = []
    for m in ["base", "mlp14", "mega"]:
        d = R["monitors"][m]
        monitors.append({"id": m, "label": names[m], "before": d["clean"]["catch"], "before_lo": d["clean"]["ci95"][0],
                         "before_hi": d["clean"]["ci95"][1], "after": d["pooled_attacked"]["catch"],
                         "after_lo": d["pooled_attacked"]["ci95"][0], "after_hi": d["pooled_attacked"]["ci95"][1],
                         "after_per_seed": d["pooled_attacked"]["per_seed"]})
    data = {"layers": layers, "monitors": monitors,
            "n": {"false_statements": R["n"]["eval_false"], "seeds": 3, "line_true": R["n"]["line_true"]}}
    plot_js = (PROJECT_DIR / "plot.js").read_text()
    print("wrote", figure.save(PROJECT_DIR, FIG, data, plot_js))


if __name__ == "__main__":
    main()
