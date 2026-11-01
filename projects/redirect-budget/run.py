"""Regenerate the post's numbers and figure from the saved captures (see README.md for the full rebuild).

    env/.venv/bin/python kran-research/projects/redirect-budget/run.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

for step in ("make_carriers.py", "bounds.py", "compare.py", "proof_matrix.py", "figure_data.py"):
    print("==", step, flush=True)
    subprocess.run([sys.executable, str(HERE / step)], check=True, cwd=HERE)
