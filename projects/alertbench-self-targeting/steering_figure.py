"""Build the causal steering figure from the pilot's steering_summary.json.

Run AFTER the steering pipeline finishes (data/steering_run/steering_summary.json):
    uv run python projects/alertbench-self-targeting/steering_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

from lib import figure

PROJECT_DIR = Path(__file__).parent
RUN = PROJECT_DIR / "data" / "steering_run"

LABELS = {
    "baseline_self": "Self (baseline)",
    "steered_self": "Self − self-direction (steered)",
    "placebo_self": "Self + random vector (placebo)",
    "baseline_other": "Other (baseline)",
}
ORDER = ["baseline_self", "steered_self", "placebo_self", "baseline_other"]


def main() -> None:
    summary = json.loads((RUN / "steering_summary.json").read_text())
    conditions = []
    for key in ORDER:
        if key in summary:
            s = summary[key]
            conditions.append({
                "key": key, "label": LABELS[key],
                "mean": s["mean"], "lo": s["ci_low"], "hi": s["ci_high"], "n": s["n"],
            })

    bs = summary.get("baseline_self", {}).get("mean")
    ss = summary.get("steered_self", {}).get("mean")
    bo = summary.get("baseline_other", {}).get("mean")
    rescue = round((ss - bs) / max(bo - bs, 1e-6) * 100, 1) if None not in (bs, ss, bo) else None

    data = {
        "conditions": conditions,
        "rescue_pct": rescue,
        "alpha": -3.0,
        "model": "Qwen2.5-0.5B-Instruct",
    }
    fig = figure.save(PROJECT_DIR, "self-targeting-steering", data,
                      (PROJECT_DIR / "plot-steering.js").read_text())
    print("wrote", fig)
    print(f"rescue toward other-baseline: {rescue}%  "
          f"(self={bs}, steered={ss}, other={bo})")


if __name__ == "__main__":
    main()
