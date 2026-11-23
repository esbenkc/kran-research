#!/bin/bash
# Pull the laptop jobs' committed outputs and rebuild every number: analysis.json, proof-matrix.md, the figure data.
set -e
cd "$(dirname "$0")"
git pull -q --rebase --autostash
uv run python test_analysis.py | tail -1
uv run python analysis.py
uv run python proof_matrix.py
uv run python figures/figure_data.py
