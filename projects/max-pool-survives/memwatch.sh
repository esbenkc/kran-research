#!/bin/bash
# Compute-gate watchdog (1018: Vera #2731; reused 1019 for clean_tokens.py + attack.py): every 15 s log the attack job's phys_footprint and system free %;
# kill the attack job if system free < 20% or footprint > 9.0 GB. Log: data/memwatch.log
LOG="$(dirname "$0")/data/memwatch.log"
while true; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub(/%/,"",$2); print $2}')
  for p in $(pgrep -f "python (attack.py --lam|clean_tokens.py)"); do
    fp=$(footprint "$p" 2>/dev/null | awk '/Footprint:/ {for(i=1;i<=NF;i++) if($i=="Footprint:") print $(i+1)}')
    echo "$(date '+%H:%M:%S') pid=$p footprint_mb=$fp free_pct=$free" >> "$LOG"
    if [ -n "$fp" ] && { [ "$fp" -gt 9000 ] || [ "$free" -lt 20 ]; }; then
      echo "$(date '+%H:%M:%S') KILL pid=$p (footprint $fp MB, free $free%)" >> "$LOG"
      kill "$p"
    fi
  done
  sleep 15
done
