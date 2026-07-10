"""Cross-model probe figure: does self-recognition scale with capability?

Reads best_acc from each Qwen probe.json across the size ladder and builds one
figure: peak "is this about me?" probe accuracy vs model size.

    python3 projects/alertbench-self-targeting/crossmodel_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # repo root for lib/
from lib import figure  # noqa: E402

PROJECT_DIR = Path(__file__).parent
DATA = PROJECT_DIR / "data"

# (params_B, label, probe-dir)
LADDER = [
    (0.5, "Qwen2.5 0.5B", "steering_run"),
    (1.5, "Qwen2.5 1.5B", "steering_qwen2-5-1-5b"),
    (3.0, "Qwen2.5 3B", "steering_qwen2-5-3b"),
    (7.0, "Qwen2.5 7B", "steering_qwen2-5-7b"),
]


def main() -> None:
    points, chance = [], None
    for size, label, sub in LADDER:
        p = json.loads((DATA / sub / "probe.json").read_text())
        chance = p.get("chance", chance)
        points.append({
            "size": size,
            "label": label,
            "acc": round(p["best_acc"], 4),
            "layer": p["best_layer"],
        })
        print(f"{label:14s} acc={p['best_acc']:.4f}  best_layer={p['best_layer']}")

    data = {"points": points, "chance": chance or 0.75, "family": "Qwen 2.5"}
    fig = figure.save(PROJECT_DIR, "self-recognition-scaling", data,
                      (PROJECT_DIR / "plot-crossmodel.js").read_text())
    print("wrote", fig)


if __name__ == "__main__":
    main()
