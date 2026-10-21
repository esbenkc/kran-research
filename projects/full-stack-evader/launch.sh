#!/bin/bash
# Launch ONE heavy job detached with a PID-bound footprint watchdog (cap 9000 MB, Vera #4311).
# Usage: ./launch.sh <logname> <python args...>     -> data/logs/<logname>.log, data/logs/memwatch_<logname>.log
cd "$(dirname "$0")"
mkdir -p data/logs
name=$1; shift
nohup ../../.venv/bin/python "$@" > data/logs/$name.log 2>&1 &
pid=$!
nohup ./watchdog.sh $pid 9000 data/logs/memwatch_$name.log > /dev/null 2>&1 &
echo "pid=$pid"
