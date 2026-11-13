#!/bin/sh
# Thought-drift number (DEVIATIONS D2, Vera #21003), alone after the lane and Owain's twin job.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
while pgrep -f "decide5.sh|after_lane.sh" >/dev/null; do sleep 3; done
echo "drift start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/drift.log -- $PY drift.py --end data/runs/d010_s0/adapter_1500.pt --n 256 --out data/reads/drift_d010_s0_1500.json
echo "drift end rc=$? $(date '+%F %T')"
