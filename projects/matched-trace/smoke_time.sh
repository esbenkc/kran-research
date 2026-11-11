#!/bin/bash
# S1/S2 timing + RSS smokes (Vera compute gate #19711), serial. PROF=1 prints per-phase times.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
COMMON="--arm joint --plant-tok 14656 --cue-delta 0.10 --plant-build leaky --k 64 --seed 0 --lr 3e-4 --kl 0.1 --kl-prefix 0.1 --chunk 8 --init data/runs/pilot/adapter.pt --dec-bias -2.255 --split-clip --time-only 10"
PROF=1 $PY watchdog.py --cap-gb 9.0 --log logs/smoke_p8.log -- $PY rl.py $COMMON --P 8 --G 8 --name smoke_p8
PROF=1 $PY watchdog.py --cap-gb 9.0 --log logs/smoke_p12.log -- $PY rl.py $COMMON --P 12 --G 8 --name smoke_p12
