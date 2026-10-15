"""Reproduce everything for "Can a free weight rescaling make removed knowledge harder to relearn?"

    uv run python projects/gauge-relearning/run.py            # all runs (~4-6 h on an M4 Mac mini)
    uv run python projects/gauge-relearning/run.py --figure   # only rebuild data/ and the figure

Steps (each is resumable: finished runs are skipped):
  1. exp.py seed 0, full lr grid per (attacker, c)        -> results/raw/seed0/
  2. exp.py --floor: never-knew-A control at c = 1         -> results/raw/seed0/floor_*
  3. exp.py seeds 1, 2 at the seed-0 best lr               -> results/raw/seed{1,2}/
  4. collect.py -> data/runs.jsonl + data/meta.json
  5. make_figure.py -> figures/gauge-recovery/{data.json, plot.js}
No model API calls: API spend is $0.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ATT = ["sgd", "adamw", "renorm", "kfac"]


def py(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    if "--figure" not in sys.argv:
        py(HERE / "exp.py", "--seeds", "0", "--attackers", *ATT)
        py(HERE / "exp.py", "--floor", "--attackers", *ATT)
        py(HERE / "exp.py", "--seeds", "1", "2", "--attackers", *ATT, "--best-only")
    py(HERE / "collect.py")
    py(HERE / "make_figure.py")


if __name__ == "__main__":
    main()
