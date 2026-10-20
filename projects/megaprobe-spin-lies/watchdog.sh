#!/bin/bash
# Footprint watchdog: log phys_footprint of PID every 10 s, kill it above KILL_MB (Vera's compute gate).
#   watchdog.sh <pid> <kill_mb> <logfile>
PID=$1; KILL_MB=$2; LOG=$3
while kill -0 "$PID" 2>/dev/null; do
  MB=$(footprint "$PID" 2>/dev/null | awk '/Footprint:/ {for(i=1;i<NF;i++) if($i=="Footprint:"){v=$(i+1); u=$(i+2)}; if (u ~ /GB/) v*=1024; if (u ~ /KB/) v/=1024; print int(v); exit}')
  echo "$(date '+%H:%M:%S') pid=$PID footprint_mb=${MB:-?} free=$(memory_pressure | awk '/free percentage/ {print $5}')" >> "$LOG"
  if [ -n "$MB" ] && [ "$MB" -gt "$KILL_MB" ]; then echo "$(date '+%H:%M:%S') KILL at ${MB} MB" >> "$LOG"; kill "$PID"; fi
  sleep 10
done
