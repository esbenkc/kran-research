#!/bin/bash
# Control cell: the same joint TEST read on the START adapter (data/runs/pilot/adapter.pt), R4, after run_main.sh has
# exited (serial lane: one heavy job at a time). Cut by the clock if it cannot start by 21:05 PST.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
while pgrep -f run_main.sh > /dev/null; do sleep 20; done
if [ $(date +%H%M) -le 2105 ]; then
  echo "start control read $(date '+%F %T')"
  $PY watchdog.py --cap-gb 9.0 --log logs/read_start_test.log -- $PY read.py --policy data/runs/pilot/adapter.pt \
      --splits test --R 4 --k 64 --B 32 --dec-bias -2.255 --plant-tok 14656 --cue-delta 0.10 --joint \
      --out data/reads/raw/start_test
  echo "end control read rc=$? $(date '+%F %T')"
else
  echo "control read cut by the clock $(date '+%F %T')"
fi
