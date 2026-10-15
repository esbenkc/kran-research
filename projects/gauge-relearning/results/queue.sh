#!/bin/bash
# Serial run queue after the seed-0 sgd/adamw/renorm grid (one MPS job at a time).
cd "$(dirname "$0")/../../.."
L=projects/gauge-relearning/results/logs
E=projects/gauge-relearning/exp.py
while pgrep -f "exp.py --seeds 0 --attackers sgd adamw renorm" >/dev/null; do sleep 15; done
while [ ! -f projects/gauge-relearning/results/raw/seed2/meta.json ]; do sleep 15; done
uv run python $E --floor --attackers sgd adamw renorm > $L/floor_sar.log 2>&1
uv run python $E --seeds 1 2 --attackers sgd adamw renorm --best-only > $L/seeds12_sar.log 2>&1
uv run python $E --seeds 0 --attackers kfac --log2c 0 4 -4 2 -2 > $L/seed0_kfac.log 2>&1
uv run python $E --floor --attackers kfac > $L/floor_kfac.log 2>&1
uv run python $E --seeds 1 2 --attackers kfac --best-only > $L/seeds12_kfac.log 2>&1
echo QUEUE DONE > $L/queue.done
