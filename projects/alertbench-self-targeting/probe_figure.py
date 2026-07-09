"""Build the mechanistic probe figure from the existing Qwen2.5-0.5B interp results.

The heavy compute already ran (in the alertbench interp pipeline); this just turns the saved
probe accuracies into the interactive figure. Source data is vendored into data/qwen_probe.json
so this repo is self-contained.

    uv run python projects/alertbench-self-targeting/probe_figure.py   # no API needed
"""
from __future__ import annotations

import json
from pathlib import Path

from lib import figure

PROJECT_DIR = Path(__file__).parent


def main() -> None:
    src = json.loads((PROJECT_DIR / "data" / "qwen_probe.json").read_text())
    points = []
    for layer_str, acc in sorted(src["layer_acc"].items(), key=lambda kv: int(kv[0])):
        lo, hi = src["layer_acc_ci"][layer_str]
        points.append({"layer": int(layer_str), "acc": acc, "lo": lo, "hi": hi})

    data = {
        "points": points,
        "chance": src["chance"],
        "best_layer": src["best_layer"],
        "best_acc": src["best_acc"],
        "n": src["n_examples"],
        "model": "Qwen2.5-0.5B-Instruct",
    }
    fig = figure.save(PROJECT_DIR, "self-other-probe", data,
                      (PROJECT_DIR / "plot-probe.js").read_text())
    print("wrote", fig)
    print(f"best layer {data['best_layer']}: {data['best_acc']:.2%} vs {data['chance']:.0%} chance (n={data['n']})")


if __name__ == "__main__":
    main()
