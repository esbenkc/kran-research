#!/bin/bash
# After the smoke: compute the v1.3 s6 cap (cap.py) and append R1 (spread decoy @1e4) and R2 (lie @1e4) to the queue.
cd "$(dirname "$0")"
PY=$HOME/work/theory-1028-instructed-lie/venv/bin/python
BATCH="--pairs 4 --klb 4 --retain 2 --deadline 10:50"   # D8 (Felix #9340): half batch; cut at the last refit before 10:50
f=data/runs/smoke_lie_spread__lam10000__seed99.json
[ -f "$f" ] && [ "$f" -nt data/logs/smoke_v13.log ] || { echo "no fresh smoke json"; exit 1; }
$PY cap.py $f > data/logs/cap.json 2>/dev/null || exit 1
cap=$(python3 -c "import json; print(json.load(open('data/logs/cap.json'))['cap'] or 300)")
until mkdir data/queue.lock 2>/dev/null; do sleep 0.2; done
{ echo "lie_spread 1e4 0 --steps $cap --fit-cities 160 $BATCH"; echo "decoy_spread 1e4 0 --steps $cap --fit-cities 160 $BATCH"; cat data/queue.txt; } > data/queue.tmp
mv data/queue.tmp data/queue.txt; rmdir data/queue.lock
echo "queued R1 R2 at cap $cap: $(cat data/logs/cap.json)"
