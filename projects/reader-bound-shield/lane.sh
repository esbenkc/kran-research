#!/bin/bash
# Serial heavy lane: runs line i of the queue file, then i+1, ...; waits (polls) for new lines. Never edit run lines;
# append only. Each line is one shell command (normally watchdog.py --cap-gb 9.0 -- <job>).
#   ./lane.sh queue.txt     (run as a named service so it outlives the tool call)
cd "$(dirname "$0")"
Q=$1; i=${2:-1}
echo "lane: start $Q at line $i $(date '+%F %T')"
while true; do
  line=$(sed -n "${i}p" "$Q")
  if [ -z "$line" ]; then sleep 10; continue; fi
  echo "[$(date '+%F %T')] start $i: $line"
  bash -c "$line"
  echo "[$(date '+%F %T')] end $i rc=$?"
  i=$((i+1))
done
