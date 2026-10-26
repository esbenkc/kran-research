#!/bin/bash
# chain_wait.sh <pid> <chain args...>: the first chain's bash died at launch (tool session ended) while s0's attack
# (pid) kept running as an orphan; wait for that pid, then run the chain (run1 skips the finished attack via its json).
cd "$(dirname "$0")"
pid=$1; shift
[ -n "$pid" ] && kill -0 "$pid" || { echo "no pid $pid"; exit 1; }
echo "$(date +%H:%M:%S) waiting for pid $pid"
while kill -0 "$pid" 2>/dev/null; do sleep 5; done
echo "$(date +%H:%M:%S) pid $pid ended"
exec ./run_chain.sh "$@"
