#!/bin/bash
# phase-0 refill timing: one process per config so each watchdog peak is per config
PY=/Users/esben/work/oracle-1105-push-lift-necessity/env/.venv/bin/python
cd "$(dirname "$0")"
for cfg in "1 32" "2 32" "4 32" "8 16"; do
  set -- $cfg
  $PY watchdog.py --cap-gb 3.4 --log logs/time_c$1_n$2.log -- $PY measure0.py --time --models cln_s0=raw/runs/cln_s0_merged.pt --chunks $1 --nfit $2 --steps 15 >> logs/time_c$1_n$2.log 2>&1
done
