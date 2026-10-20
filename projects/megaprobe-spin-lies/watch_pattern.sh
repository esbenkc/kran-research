#!/bin/bash
# Follow whichever python process matches PATTERN (a chain of jobs), log its footprint every 10 s, kill above KILL_MB.
# Exits after 120 s with no matching process.   watch_pattern.sh <pattern> <kill_mb> <logfile>
PAT=$1; KILL_MB=$2; LOG=$3; idle=0
while [ $idle -lt 12 ]; do
  PID=$(pgrep -f "\.venv/bin/python .*$PAT" | tail -1)
  if [ -z "$PID" ]; then idle=$((idle+1)); sleep 10; continue; fi
  idle=0
  MB=$(footprint "$PID" 2>/dev/null | awk '/Footprint:/ {for(i=1;i<NF;i++) if($i=="Footprint:"){v=$(i+1); u=$(i+2)}; if (u ~ /GB/) v*=1024; if (u ~ /KB/) v/=1024; print int(v); exit}')
  echo "$(date '+%H:%M:%S') pid=$PID footprint_mb=${MB:-?} free=$(memory_pressure | awk '/free percentage/ {print $5}')" >> "$LOG"
  if [ -n "$MB" ] && [ "$MB" -gt "$KILL_MB" ]; then echo "$(date '+%H:%M:%S') KILL at ${MB} MB" >> "$LOG"; kill "$PID"; fi
  sleep 10
done
