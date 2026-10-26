#!/bin/bash
# Give a teammate an exclusive slot between chain jobs: wait for the running chain to exit (a 'stop' line in
# queue.txt), touch the flag file, wait until the teammate deletes it (or the cap passes), restart the chain.
#   ./slot.sh <chain_pid> <flag_path> <cap_seconds>
cd "$(dirname "$0")"
while kill -0 $1 2>/dev/null; do sleep 2; done
mkdir -p "$(dirname "$2")"; touch "$2"
echo "$(date '+%H:%M:%S') SLOT_OPEN $2 (cap $3 s)" >> logs/chain.log
t=0; while [ -e "$2" ] && [ $t -lt $3 ]; do sleep 5; t=$((t+5)); done
rm -f "$2"
echo "$(date '+%H:%M:%S') SLOT_CLOSED $2 after $t s" >> logs/chain.log
exec ./chain2.sh
