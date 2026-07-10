"""Build the elicitation figure from reanalysis.json.

    python3 projects/alertbench-self-targeting/elicitation_figure.py [key]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # repo root for lib/
from lib import figure  # noqa: E402

PROJECT_DIR = Path(__file__).parent


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen2-5-3b-instruct"
    rean = json.loads((PROJECT_DIR / "data" / f"elicitation_{key}" / "reanalysis.json").read_text())
    rows = [r for r in rean["by_framing"] if r["self_mean"] is not None]
    data = {"rows": rows, "model": "Qwen2.5-3B-Instruct"}
    fig = figure.save(PROJECT_DIR, "elicitation-framing", data,
                      (PROJECT_DIR / "plot-elicitation.js").read_text())
    print("wrote", fig)
    for r in rows:
        print(f"  {r['framing']:14s} self={r['self_mean']} other={r['other_mean']} "
              f"gap={r['delta']} p={r['p']}")


if __name__ == "__main__":
    main()
