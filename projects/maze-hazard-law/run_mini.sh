#!/usr/bin/env bash
# Mini side row: one agent per process (<= 2B params), gate first, then maze + dead end if gated (PREREG).
# Usage (cwd = projects/maze-hazard-law): bash run_mini.sh <model> <maze n> <deadend n> <unix deadline>
set -euo pipefail
MODEL="$1"; NMAZE="$2"; NDEAD="$3"; DEADLINE="$4"
OUT="data/$MODEL"
export HF_HUB_OFFLINE=1
.venv/bin/python web/run.py --model "$MODEL" --n visible=10 --out "$OUT" --deadline "$DEADLINE"
.venv/bin/python web/run.py --model "$MODEL" --n "maze=$NMAZE,deadend=$NDEAD" --out "$OUT" --require-gate 8 --deadline "$DEADLINE"
