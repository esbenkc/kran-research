#!/bin/bash
# One RL lane: runs the queue file lanes/<lane>.q line by line (popped under a mkdir lock, so lines can be added or
# removed while the lane runs). Each line = the rl.py arguments. Every job runs under watchdog.py (footprint cap).
#   ./lane.sh A      (start from the project dir's parent of parents: ~/work/oracle-1030-stack-conditional)
set -u
L=$1
P=kran-research/projects/stack-conditional
PY=env/.venv/bin/python
Q=$P/lanes/$L.q
CAP=${CAP:-2.8}
while true; do
  until mkdir $P/lanes/.lock 2>/dev/null; do sleep 1; done
  line=$(head -n 1 "$Q" 2>/dev/null)
  if [ -n "$line" ]; then tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"; fi
  rmdir $P/lanes/.lock
  if [ -z "$line" ]; then
    [ -e $P/lanes/$L.stop ] && { echo "$(date '+%F %T') lane $L stop" >> $P/logs/lane_$L.log; exit 0; }
    sleep 10; continue
  fi
  tag=$(echo "$line" | tr ' ' '_' | tr -d '-')
  echo "$(date '+%F %T') lane $L START $line" >> $P/logs/lane_$L.log
  $PY $P/watchdog.py --cap-gb $CAP --log $P/logs/rl_$tag.log -- $PY $P/rl.py $line
  echo "$(date '+%F %T') lane $L END rc=$? $line" >> $P/logs/lane_$L.log
done
