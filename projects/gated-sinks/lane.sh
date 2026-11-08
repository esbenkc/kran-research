#!/bin/bash
# Job lane: runs the lines of queue file $1 in order, one at a time, waiting for new lines to be
# appended; stops at a line 'END'. Logs 'start/end <line> rc=' to logs/lane_<name>.log.
# Never edit this file while a lane runs (bash reads scripts by offset); append to the queue instead.
cd "$(dirname "$0")"
q=$1
name=$(basename "$q" .txt)
i=1
while true; do
  line=$(sed -n "${i}p" "$q")
  if [ -z "$line" ]; then sleep 10; continue; fi
  if [ "$line" = "END" ]; then echo "lane $name done" >> logs/lane_$name.log; exit 0; fi
  echo "$(date +%H:%M:%S) start $line" >> logs/lane_$name.log
  bash -c "$line"
  echo "$(date +%H:%M:%S) end rc=$? $line" >> logs/lane_$name.log
  i=$((i + 1))
done
