#!/bin/bash
# Serial training queue (one heavy job at a time; DEVIATIONS.md D1 order). Run from projects/probe-ensemble/.
#   JUDGE_MODEL=gemini-3-flash-preview with-model-keys ./run_queue.sh <wait_pid> "ensemble 1 0" "single 1 1" ...
# Each job is "penalty lam seed". A job does not start after LAST_START (local time HHMM; default 05:05 CEST =
# 20:05 PDT on this machine), so training ends by the 05:30 CEST stop.
set -u
WAIT_PID=$1; shift
LAST_START=${LAST_START:-2005}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
for job in "$@"; do
  set -- $job
  pen=$1; lam=$2; seed=$3
  now=$(date +%H%M)
  if [ "$now" -gt "$LAST_START" ]; then echo "$(date +%T) SKIP $job (after $LAST_START)"; continue; fi
  name=$([ "$pen" = ensemble ] && echo "ens_")lam${lam}_s${seed}
  echo "$(date +%T) START $name"
  uv run python -u train.py --penalty "$pen" --lam "$lam" --seed "$seed" > "data/logs/train_${name}.log" 2>&1
  echo "$(date +%T) END $name exit=$?"
done
