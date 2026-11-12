#!/bin/bash
# One lane: re-read the queue file before every claim, claim the first unclaimed job (atomic mkdir raw/claims/<name>),
# run it under watchdog.py (2.5 GB cap). The queue is DATA, so lines can be appended or reordered while the lane runs.
# Line format: <name>|<rl.py args>      (# = comment). Idle -> poll every 15 s. Stop: touch raw/claims/STOP.
# usage: ./lane.sh A queue.txt
cd "$(dirname "$0")"
LANE=$1; Q=$2; PY=../../../env/.venv/bin/python
mkdir -p raw/claims logs
while true; do
  [ -f raw/claims/STOP ] && { echo "$(date +%T) lane $LANE: STOP"; exit 0; }
  picked=""
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    name=${line%%|*}; args=${line#*|}
    [ -d "raw/claims/$name" ] && continue
    if mkdir "raw/claims/$name" 2>/dev/null; then picked=$name; break; fi
  done < "$Q"
  if [ -z "$picked" ]; then sleep 15; continue; fi
  echo "$(date +%T) lane $LANE: start $picked :: $args"
  echo "$PY rl.py --name $picked $args" > "logs/$picked.cmd"
  $PY watchdog.py --cap-gb 2.5 --log "logs/$picked.log" -- $PY rl.py --name "$picked" $args > /dev/null 2>&1
  rc=$?
  echo "$(date +%T) lane $LANE: end $picked exit $rc :: $(grep -E '^DONE|WATCHDOG: exit' logs/$picked.log | tr '\n' ' ' | cut -c1-200)"
  touch "logs/$picked.done"
done
