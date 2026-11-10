#!/bin/bash
# One lane: repeatedly re-read the queue file, claim the first unclaimed job (atomic mkdir raw/claims/<name>), run it.
# The queue is DATA re-read before every claim, so lines can be appended or reordered while lanes run.
# Line format: <name>|<gate>|<rl.py args>   gate = none | p0 (needs phase0_gate.py exit 0; exit 1 = Phase 0 failed -> skip)
# usage: ./lane.sh A queue.txt
cd "$(dirname "$0")"
LANE=$1; Q=$2; PY=../../../env/.venv/bin/python
while true; do
  [ -f raw/claims/STOP ] && { echo "$(date +%T) lane $LANE: STOP"; exit 0; }
  picked=""
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    name=${line%%|*}; rest=${line#*|}; gate=${rest%%|*}; args=${rest#*|}
    [ -d "raw/claims/$name" ] && continue
    if [ "$gate" = "p0" ]; then
      /usr/bin/python3 phase0_gate.py >/dev/null; g=$?
      [ $g -eq 2 ] && continue          # Phase 0 still pending: skip for now
      [ $g -eq 1 ] && { mkdir -p "raw/claims/$name"; echo "$(date +%T) lane $LANE: $name skipped (Phase 0 failed)"; continue; }
    fi
    if mkdir "raw/claims/$name" 2>/dev/null; then picked=$name; break; fi
  done < "$Q"
  if [ -z "$picked" ]; then
    # nothing claimable: done if no gated line is pending, else wait
    pending=$(grep -v '^#' "$Q" | grep -c . ); claimed=$(ls raw/claims | grep -vc STOP)
    [ "$claimed" -ge "$pending" ] && { echo "$(date +%T) lane $LANE: queue empty"; exit 0; }
    sleep 20; continue
  fi
  echo "$(date +%T) lane $LANE: start $picked :: $args"
  /usr/bin/time -l $PY watchdog.py --cap-gb 2.5 --log logs/$picked.log -- $PY rl.py --name $picked $args >> logs/$picked.time 2>&1
  echo "$(date +%T) lane $LANE: end $picked exit $?"
  touch logs/$picked.done
done
