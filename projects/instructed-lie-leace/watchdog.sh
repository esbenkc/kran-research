#!/bin/bash
# Footprint watchdog bound to ONE pid (never a pgrep pattern): logs phys_footprint every 5 s, kills the pid above a cap.
# Usage: ./watchdog.sh <pid> <cap_mb> <logfile>
pid=$1; cap=$2; log=$3
while kill -0 "$pid" 2>/dev/null; do
  mb=$(footprint "$pid" 2>/dev/null | awk '/Footprint:/ {for (i=1;i<NF;i++) if ($i=="Footprint:") {v=$(i+1); u=$(i+2)}; if (u=="GB") v*=1024; if (u=="KB") v/=1024; print int(v); exit}')
  echo "$(date +%H:%M:%S) pid=$pid footprint_mb=${mb:-NA}" >> "$log"
  if [ -n "$mb" ] && [ "$mb" -gt "$cap" ]; then
    echo "$(date +%H:%M:%S) KILL pid=$pid footprint_mb=$mb > cap $cap" >> "$log"; kill "$pid"; exit 1
  fi
  sleep 5
done
