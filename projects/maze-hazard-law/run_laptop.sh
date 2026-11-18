#!/usr/bin/env bash
# Laptop-queue job for the status agents (gpt-oss-20b-low, qwen3-8b-nothink). One command, resumable:
# run.py skips every run whose raw/<arm>/<i>.json exists, so a later job on a commit that holds this job's pushed
# outputs continues where it stopped.
# Usage (cwd = projects/maze-hazard-law): bash run_laptop.sh <gpt-oss-20b-low|qwen3-8b-nothink> <awake hours> <maze n> <deadend n>
# Order: the 10-run visible-link gate first, then maze and dead-end runs interleaved by run index (PREREG).
# The budget counts AWAKE time (macOS CLOCK_UPTIME_RAW), so a sleeping lid does not eat it.
set -euo pipefail
MODEL="$1"
HOURS="${2:-2.5}"
NMAZE="${3:-40}"
NDEAD="${4:-20}"
OUT="data/$MODEL"
AWAKE_DEADLINE=$(python3 -c "import time; print(time.clock_gettime(time.CLOCK_UPTIME_RAW) + $HOURS * 3600)")
uv sync --frozen
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
caffeinate -i uv run python web/run.py --model "$MODEL" --n visible=10 --out "$OUT" --awake-deadline "$AWAKE_DEADLINE"
caffeinate -i uv run python web/run.py --model "$MODEL" --n "maze=$NMAZE,deadend=$NDEAD" --out "$OUT" --require-gate 8 --awake-deadline "$AWAKE_DEADLINE"
