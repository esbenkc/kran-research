#!/bin/bash
# heavy.sh <logname> <script.py> [args...]: ONE heavy job + its PID-bound footprint watchdog (kill at 9000 MB).
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1028-instructed-lie/venv/bin/python}
log=$1; shift
# MPS allocator caps (D10): free cached blocks early so the phys_footprint stays under the 9000 MB watchdog
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.65}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}
[ -f data/logs/$log.log ] && mv data/logs/$log.log data/logs/$log.before$(date +%H%M%S).log
$PY "$@" > data/logs/$log.log 2>&1 &
pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_$log.log &
wait $pid
