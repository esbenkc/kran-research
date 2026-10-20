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
        layers.append({"layer": l, "before": d["clean"], "before_lo": d["clean_ci95"][0], "before_hi": d["clean_ci95"][1],
                       "after": d["attacked"], "after_lo": d["attacked_ci95"][0], "after_hi": d["attacked_ci95"][1],
                       "after_per_seed": d["per_seed"], "kept": d["retention"], "kept_lo": d["retention_ci95"][0],
                       "kept_hi": d["retention_ci95"][1], "scored": d["scored"]})
    D3 = json.loads((PROJECT_DIR / "data" / "results" / "d3.json").read_text())  # DEVIATIONS D3, exploratory
    for d in layers:
        m = R["monitors"][f"lin{d['layer']}"]
        t = D3["tok"][str(d["layer"])]["catch"]
        d.update({"mean_catch_before": m["clean"]["catch"], "mean_catch_after": m["pooled_attacked"]["catch"],
                  "mean_catch_lo": m["pooled_attacked"]["ci95"][0], "mean_catch_hi": m["pooled_attacked"]["ci95"][1],
                  "tok_catch_before": t["clean"]["catch"], "tok_catch_after": t["pooled_attacked"]["catch"],
                  "tok_catch_lo": t["pooled_attacked"]["ci95"][0], "tok_catch_hi": t["pooled_attacked"]["ci95"][1],
                  "tok_false_alarms": [t[st]["false_alarm_eval"] for st in ("seed0", "seed1", "seed2")]})
    names = {"base": "one-layer probe (layer 14)", "mlp14": "small neural probe (layer 14)", "mega": "megaprobe (every layer)"}
    monitors = []
    for m in ["base", "mlp14", "mega"]:
        d = R["monitors"][m]
        monitors.append({"id": m, "label": names[m], "before": d["clean"]["catch"], "before_lo": d["clean"]["ci95"][0],
                         "before_hi": d["clean"]["ci95"][1], "after": d["pooled_attacked"]["catch"],
                         "after_lo": d["pooled_attacked"]["ci95"][0], "after_hi": d["pooled_attacked"]["ci95"][1],
                         "after_per_seed": d["pooled_attacked"]["per_seed"]})
    data = {"layers": layers, "monitors": monitors,
            "n": {"false_statements": R["n"]["eval_false"], "seeds": 3}}
    plot_js = (PROJECT_DIR / "plot.js").read_text()
    print("wrote", figure.save(PROJECT_DIR, FIG, data, plot_js))


if __name__ == "__main__":
    main()
