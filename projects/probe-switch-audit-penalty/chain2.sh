#!/bin/bash
# Runs jobs one at a time from queue.txt (top line first; edit the file to re-order while the chain runs).
# Line format:  train <name> <train_pen.py args...>  |  whole <run1,run2,...>  |  audit <name>  |  stop
# Every job runs under watchdog.py (footprint cap bound to the child pid); one heavy job at a time.
cd "$(dirname "$0")"
P=$(pwd)
PY=/Users/esben/work/oracle-1026-audit-penalty/env/.venv/bin/python
Q=$P/queue.txt
log() { echo "$(date '+%H:%M:%S') $*" >> $P/logs/chain.log; }
pop() {  # pop the first line under a mkdir lock
  while ! mkdir $P/.qlock 2>/dev/null; do sleep 1; done
  LINE=$(head -1 $Q); tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q
  rmdir $P/.qlock
}
while true; do
  pop
  [ -z "$LINE" ] && { log "queue empty, waiting"; sleep 20; continue; }
  set -- $LINE
  kind=$1; shift
  case $kind in
    stop) log "STOP"; exit 0 ;;
    train) name=$1; shift
      log "START train $name $*"
      $PY watchdog.py --cap-gb 6.0 --log logs/train_$name.log -- $PY train_pen.py --name $name "$@"
      log "DONE train $name rc=$?" ;;
    whole) log "START whole $1"
      $PY watchdog.py --cap-gb 6.0 --log logs/whole_$(date +%H%M%S).log -- $PY eval_whole.py --runs $1 --emulate 0
      log "DONE whole $1 rc=$?" ;;
    audit) name=$1; shift
      log "START audit $name $*"
      $PY watchdog.py --cap-gb 5.0 --log logs/audit_$name.log -- $PY audit.py --run $name "$@"
      log "DONE audit $name rc=$?" ;;
    *) log "unknown line: $LINE" ;;
  esac
done
