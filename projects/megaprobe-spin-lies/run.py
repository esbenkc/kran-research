"""Regenerate the figure from the saved fits (the heavy steps are in data/README.md: cache.py, probes.py, fit_c2.py,
refit_b.py). This file re-scores every arm and writes figures/megaprobe-spin-lies/{data.json, plot.js}, the files
copied to blog.kran.ai/assets/figures/megaprobe-spin-lies/.

    uv run python projects/megaprobe-spin-lies/run.py        (from kran-research/; CPU, ~4 min)
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

import score  # noqa: E402
from lib import figure  # noqa: E402


def main() -> None:
    score.main()
    data = json.loads((PROJECT_DIR / "data/results/figure_data.json").read_text())
    fig = figure.save(PROJECT_DIR, "megaprobe-spin-lies", data, (PROJECT_DIR / "plot.js").read_text())
    print("wrote", fig)


if __name__ == "__main__":
    main()
