#!/bin/bash
# Scored queue, ONE heavy job at a time (PREREG s3 order): each run, then its extraction, then light panel scoring.
# Every python job gets its own PID-bound footprint watchdog (kill at 9000 MB, PREREG s11).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
run1() {  # run1 <target> <lam> <seed> [steps, default 150]
  steps=${4:-150}; tag=gap; [ "$steps" != 150 ] && tag=gap$steps
  name="$1__${tag}__lam$2__seed$3"
  if [ ! -f data/runs/$name.json ]; then
    $PY attack_full.py --target $1 --lam $2 --seed $3 --steps $steps $EXTRA > data/logs/attack_$name.log 2>&1 &
    pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_attack_$name.log &
    wait $pid || { echo "FAILED attack $name"; exit 1; }
  fi
  if [ ! -f data/acts/$name/meta.json ]; then
    $PY extract.py --state $name --adapter data/runs/${name}__adapter > data/logs/extract_$name.log 2>&1 &
    pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_extract_$name.log &
    wait $pid || { echo "FAILED extract $name"; exit 1; }
  fi
  $PY panel.py score $name > data/logs/score_$name.log 2>&1 || { echo "FAILED score $name"; exit 1; }
  echo "$(date +%H:%M:%S) DONE $name"
}
for spec in "$@"; do run1 $spec; done
echo "$(date +%H:%M:%S) CHAIN_DONE"
