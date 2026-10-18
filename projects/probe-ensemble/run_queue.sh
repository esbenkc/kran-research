#!/bin/bash
# Serial job queue (one heavy job at a time; DEVIATIONS.md D1/D2 order). Run from projects/probe-ensemble/.
#   JUDGE_MODEL=gemini-3-flash-preview with-model-keys ./run_queue.sh <wait_pid> "ensemble 1 0" "single 1 1" ... refit
# A training job is "penalty lam seed"; "refit" runs refit.py. Times are local machine time (PDT = CEST - 9 h):
#   LAST_START (HHMM, default 2005 = 05:05 CEST): no training job starts later;
#   KILL_AT    (HHMM, default 2050 = 05:50 CEST): a training job still running then is killed (logged KILLED).
set -u
WAIT_PID=$1; shift
LAST_START=${LAST_START:-2005}
KILL_AT=${KILL_AT:-2050}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
for job in "$@"; do
  if [ "$job" = refit ]; then
    echo "$(date +%T) START refit"
    (cd ../.. && uv run python -u projects/probe-ensemble/refit.py) > data/logs/refit.log 2>&1
    echo "$(date +%T) END refit exit=$?"
    continue
  fi
  set -- $job
  pen=$1; lam=$2; seed=$3
  if [ "$(date +%H%M)" -gt "$LAST_START" ]; then echo "$(date +%T) SKIP $job (after $LAST_START)"; continue; fi
  name=$([ "$pen" = ensemble ] && echo "ens_")lam${lam}_s${seed}
  echo "$(date +%T) START $name"
  uv run python -u train.py --penalty "$pen" --lam "$lam" --seed "$seed" > "data/logs/train_${name}.log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date +%H%M)" -ge "$KILL_AT" ]; then pkill -P "$pid"; kill "$pid"; echo "$(date +%T) KILLED $name at $KILL_AT"; fi
    sleep 10
  done
  wait "$pid"; echo "$(date +%T) END $name exit=$?"
done
