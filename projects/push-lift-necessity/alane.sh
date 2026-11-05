#!/bin/bash
# Post-training audit lane: run sched.py's next job until nothing is left (both P runs finished) or 20:30 PST.
cd "$(dirname "$0")"
L=$1; PY=/Users/esben/work/oracle-1105-push-lift-necessity/env/.venv/bin/python; idle=0
while true; do
  [ "$(date +%H%M)" -ge 2030 ] && { echo "ALANE $L 20:30 STOP $(date +%T)"; break; }
  out=$($PY sched.py next $L)
  if [ -z "$out" ]; then
    if [ -f data/runs/P_s0.json ] && [ -f data/runs/P_s1.json ]; then idle=$((idle+1)); fi
    [ $idle -ge 6 ] && { echo "ALANE $L EMPTY $(date +%T)"; break; }
    sleep 20; continue
  fi
  idle=0; jid=${out%%$'\t'*}; cmd=${out#*$'\t'}
  echo "ALANE $L START $jid $(date +%T)"
  eval "$cmd"
  echo "ALANE $L END $jid rc=$? $(date +%T)"
done
