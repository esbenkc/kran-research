#!/usr/bin/env bash
# Laptop-queue job for the agents the goal names (gpt-oss-20b, Qwen3-8B). One command, resumable:
# run.py skips every run whose raw/<arm>/<i>.json exists, so a restarted job continues where it stopped.
# Usage (cwd = projects/maze-cost): bash run_laptop.sh <gpt-oss-20b|qwen3-8b> <hours>
# Order: the 10-run visible-link gate first, then maze and dead-end runs interleaved by run index (PREREG s2).
set -euo pipefail
MODEL="$1"
HOURS="${2:-2.5}"
OUT="data/$MODEL"
DEADLINE=$(python3 -c "import time; print(time.time() + $HOURS * 3600)")
uv sync --frozen
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
caffeinate -i uv run python web/run.py --model "$MODEL" --n visible=10 --out "$OUT" --deadline "$DEADLINE"
caffeinate -i uv run python web/run.py --model "$MODEL" --n maze=42,deadend=40 --out "$OUT" --deadline "$DEADLINE"
