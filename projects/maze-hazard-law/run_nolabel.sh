#!/usr/bin/env bash
# DEVIATIONS D8 (reported row only, never status): gpt-oss-20b-low on the maze with the page number removed
# (arm maze_nolabel, web/sites.py), full prefill on every call, same harness, seeds, templates, cap and sampling as
# data/gpt-oss-20b-low-full. The gate is not rerun: run.py reads the 10 gate runs of data/gpt-oss-20b-low-full
# (same model and settings). One command, resumable (run.py skips every existing raw/<arm>/<i>.json).
# Usage (cwd = projects/maze-hazard-law): bash run_nolabel.sh <awake hours> <maze_nolabel n>
set -euo pipefail
HOURS="${1:-1.25}"
N="${2:-39}"
export MAZE_PREFIX_REUSE=0
OUT="data/gpt-oss-20b-low-nolabel"
AWAKE_DEADLINE=$(python3 -c "import time; print(time.clock_gettime(time.CLOCK_UPTIME_RAW) + $HOURS * 3600)")
uv sync --frozen
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
caffeinate -i uv run python web/run.py --model gpt-oss-20b-low --n "maze_nolabel=$N" --out "$OUT" \
  --require-gate 8 --gate-dir data/gpt-oss-20b-low-full --awake-deadline "$AWAKE_DEADLINE"
