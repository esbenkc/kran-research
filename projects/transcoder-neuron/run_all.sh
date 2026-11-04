#!/bin/bash
# Full T2 run, layers sequential, one heavy job at a time, footprint cap 2.9 GB (watchdog.py).
cd "$(dirname "$0")"
for L in 8 9 10 11; do
  env/.venv/bin/python watchdog.py --cap-gb 3.1 --log logs/wd_L$L.log -- env/.venv/bin/python run.py --layer $L --tok 2048 > logs/run_L$L.log 2>&1
  tail -1 logs/wd_L$L.log
done
