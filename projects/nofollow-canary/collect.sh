#!/bin/bash
# Rebuild every number from the saved raw data (room defense-1124-length-or-markup). Run in projects/nofollow-canary.
#   bash collect.sh
set -euo pipefail
git pull -q --rebase --autostash
uv sync --frozen -q
uv run python test_analysis.py | tail -1                       # boundary checks of the frozen rule (no data)
uv run python analysis.py                                      # results/analysis.json + results/analysis_runs.csv
uv run python proof_matrix.py                                  # proof-matrix.md (skeptic rows below the marker kept)
python3 figures/build_main.py --out web/out/qwen3-8b ${BLOG:+--blog "$BLOG"}   # main figure data (Selma's own rule copy)
