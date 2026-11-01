#!/bin/bash
# One heavy job at a time from queue.txt (first line popped under a mkdir lock, so the queue can be edited while a job
# runs). Line format: <logname> <script.py> <args...>. Each job runs under watchdog.py (footprint cap, bound to the
# child's pid). When the queue is empty it polls for new lines up to IDLE seconds (default 3600), then exits.
# START/END lines with times go to logs/chain.log.
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1027-forced-attention/env/.venv/bin/python
CAP=${CAP:-8.5}
IDLE=${IDLE:-3600}
EMPTY_SINCE=""
while true; do
  until mkdir .qlock 2>/dev/null; do sleep 1; done
  LINE=$(grep -v '^[[:space:]]*$' queue.txt | grep -v '^#' | head -1)
  if [ -n "$LINE" ]; then
    grep -vxF "$LINE" queue.txt > queue.tmp; mv queue.tmp queue.txt
  fi
  rmdir .qlock
  if [ -z "$LINE" ]; then
    [ -z "$EMPTY_SINCE" ] && { EMPTY_SINCE=$(date +%s); echo "QUEUE_EMPTY $(date)" >> logs/chain.log; }
    [ $(( $(date +%s) - EMPTY_SINCE )) -gt $IDLE ] && { echo "CHAIN_EXIT idle $(date)" >> logs/chain.log; exit 0; }
    sleep 5; continue
  fi
  EMPTY_SINCE=""
  set -- $LINE
  NAME=$1; shift
  echo "START $NAME $(date) :: $*" >> logs/chain.log
  $PY watchdog.py --cap-gb $CAP --log logs/$NAME.log -- $PY "$@"
  echo "END $NAME rc=$? $(date)" >> logs/chain.log
done
