#!/bin/bash
# heavy.sh <logname> <script.py> [args...]: ONE heavy job + its PID-bound footprint watchdog (kill at 9000 MB).
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1027-refit-probe-evader/venv/bin/python}
log=$1; shift
[ -f data/logs/$log.log ] && mv data/logs/$log.log data/logs/$log.before$(date +%H%M%S).log
$PY "$@" > data/logs/$log.log 2>&1 &
pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_$log.log &
wait $pid
