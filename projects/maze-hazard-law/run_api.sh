#!/usr/bin/env bash
# API side row (DEVIATIONS D4): gpt-5-nano, gate first, then maze + dead end if gated; 4 runs in flight, one process.
# Usage (cwd = projects/maze-hazard-law): with-model-keys bash run_api.sh <unix deadline>
set -euo pipefail
DEADLINE="$1"
OUT="data/gpt-5-nano"
.venv/bin/python web/run.py --model gpt-5-nano --n visible=10 --out "$OUT" --deadline "$DEADLINE" --threads 4
.venv/bin/python web/run.py --model gpt-5-nano --n maze=40,deadend=20 --out "$OUT" --require-gate 8 --deadline "$DEADLINE" --threads 4
