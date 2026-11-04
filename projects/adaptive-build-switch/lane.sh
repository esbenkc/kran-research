#!/bin/bash
# Run lanes/<lane>.q one line at a time (re-read every loop, so the queue can be edited while it runs).
# A line = "<name> <args for rl_adapt.py>" or "<name> !<shell command>"; finished names go to lanes/<lane>.done; a 'STOP' line ends the lane.
# Each job runs under watchdog.py (footprint cap) with its own log logs/<name>.log.
cd "$(dirname "$0")"
L=$1; PY=/Users/esben/work/oracle-1104-adaptive-switch/env/.venv/bin/python
touch lanes/$L.done
while true; do
  line=$(grep -v '^#' lanes/$L.q | grep -v '^\s*$' | while read -r n rest; do grep -qx "$n" lanes/$L.done || { echo "$n $rest"; break; }; done)
  [ -z "$line" ] && { echo "LANE $L EMPTY $(date +%T)"; break; }
  name=${line%% *}; args=${line#* }
  [ "$name" = "STOP" ] && { echo "LANE $L STOP $(date +%T)"; break; }
  echo "LANE $L START $name $(date +%T)"
  if [ "${args:0:1}" = "!" ]; then  # '!<shell command>' = an eval/aux job (its own watchdog inside the command)
    (eval "${args:1}") >> logs/$name.log 2>&1
  else
    $PY watchdog.py --cap-gb ${CAPGB:-2.85} --log logs/$name.log -- $PY rl_adapt.py --name $name $args >> logs/$name.log 2>&1
  fi
  echo "$name" >> lanes/$L.done
  echo "LANE $L END $name rc=$? $(date +%T) $(grep -E '^DONE' logs/$name.log | tail -1 | cut -c1-200)"
done
