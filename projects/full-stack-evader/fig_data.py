"""Figure data: data/results/results.json (+ Iris's blind floor) -> blog assets/figures/full-stack-evader/data.json.
The texts (title, subtitle, legend) live in fig_text.json, so that figure and post use one set of words.
Run: python3 fig_data.py [path/to/blog.kran.ai]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / "data/results/results.json").read_text())
BLOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "blog.kran.ai"
OUT = BLOG / "assets/figures/full-stack-evader/data.json"
TEXT = json.loads((HERE / "fig_text.json").read_text())
# Iris #4345 (clean activations only, posted before any attacked number): at the 0.05 bar the bound allows a catch of
# 0% for the probe on the last layer after the final normalisation (readout 29, shown as "28+").
FLOOR_L28POST = 0.0


def entry(group, key, prefix):
    m = R[group]["panel"][key]
    return {prefix: m["catch"], prefix + "_lo": m["lo"], prefix + "_hi": m["hi"], prefix + "_per_seed": m["per_seed"]}


def main():
    clean = R["clean"]
    layers = []
    for r in range(1, 30):
        k = f"tok{r}"
        layers.append({"layer": r, "layer_label": "28+" if r == 29 else str(r), "clean": clean[k]["catch"],
                       **entry("a", k, "a"), **entry("b", k, "b"),
                       "c": R["control_a_lam0"]["panel"][k]["catch"],
                       "floor": FLOOR_L28POST if r == 29 else None})
    # sentence-average probe: the layer with the lowest catch under either fine-tune, among layers with clean >= 0.5
    means = [f"mean{r}" for r in range(1, 29) if clean[f"mean{r}"]["catch"] >= 0.5]
    worst = min(means, key=lambda k: min(R["a"]["panel"][k]["catch"], R["b"]["panel"][k]["catch"]))
    others = []
    for key, label in [("mega", "megaprobe (all layers)"), ("mlp14", "small neural probe (layer 14)"),
                       (worst, f"sentence-average, layer {worst[4:]}")]:
        others.append({"id": key, "label": label, "clean": clean[key]["catch"], **entry("a", key, "a"), **entry("b", key, "b")})
    data = {
        "title": TEXT["title"], "subtitle": TEXT["subtitle"], "x_label": TEXT["x_label"], "y_label": TEXT["y_label"],
        "bar": 0.2, "target_layer": 18, "legend": TEXT["legend"], "layers": layers,
        "others_title": TEXT["others_title"], "others": others,
        "n": {"seeds_a": len(R["a"]["panel"]["tok18"]["per_seed"]), "seeds_b": len(R["b"]["panel"]["mega"]["per_seed"]),
              "false_statements": 596},
    }
    OUT.write_text(json.dumps(data, indent=1))
    a, b = R["a"]["panel"], R["b"]["panel"]
    print(OUT, "| a: tok18", a["tok18"]["catch"], "tok19", a["tok19"]["catch"], "mega", a["mega"]["catch"],
          "| b: mega", b["mega"]["catch"], "min tok10-29", min(b[f"tok{r}"]["catch"] for r in range(10, 30)))


if __name__ == "__main__":
    main()
