#!/bin/bash
# Run one probes.py / analysis stage with a PID-bound footprint watchdog. Usage: ./run_stage.sh <cap_mb> <script.py> [args]
cd "$(dirname "$0")"
mkdir -p data/logs
cap=$1; shift
name=$(echo "$*" | tr ' /' '__' | sed 's/\.py//')
../../.venv/bin/python "$@" > data/logs/$name.log 2>&1 &
pid=$!
./watchdog.sh $pid $cap data/logs/memwatch_$name.log &
wait $pid; rc=$?
grep -v -i "warn" data/logs/$name.log | tail -40
echo "peak_footprint_mb $(awk -F'footprint_mb=' '{print $2}' data/logs/memwatch_$name.log | sort -n | tail -1) rc=$rc"
exit $rc
