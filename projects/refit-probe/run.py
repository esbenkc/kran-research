"""refit-probe: does a probe refit on the trained model's own activations catch the lies the old probe misses?

Pipeline (from kran-research/), no training; the policies are saved round-8 LoRA adapters of earlier runs:
    uv run python projects/refit-probe/gen.py            # answers + activations, ~75 min on an M4 (MPS), ~3 GB
    with-model-keys uv run python projects/refit-probe/label.py   # judge + second judge + string check (cached)
    uv run python projects/refit-probe/run.py            # refits, verdict, figure data (CPU, a few minutes)
This file reruns the analysis from the saved answers, labels and activations in data/ and exports the figure.
"""
from pathlib import Path

import refit

PROJECT_DIR = Path(__file__).parent

if __name__ == "__main__":
    refit.main()
    print("figure data:", PROJECT_DIR / "data/results/figure_data.json")
