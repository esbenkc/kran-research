#!/bin/bash
# K-FAC: seed-0 grid split over 3 processes (cells claim a .tmp file), then floor, then seeds 1,2.
cd "$(dirname "$0")/../../.."
L=projects/gauge-relearning/results/logs
E=projects/gauge-relearning/exp.py
uv run python $E --seeds 0 --attackers kfac --log2c 0 4 > $L/seed0_kfac_a.log 2>&1 &
sleep 60
uv run python $E --seeds 0 --attackers kfac --log2c -4 > $L/seed0_kfac_b.log 2>&1 &
sleep 60
uv run python $E --seeds 0 --attackers kfac --log2c 2 -2 > $L/seed0_kfac_c.log 2>&1 &
wait
uv run python $E --seeds 0 --attackers kfac > $L/seed0_kfac_final.log 2>&1   # edge extensions, if any
uv run python $E --floor --attackers kfac > $L/floor_kfac.log 2>&1
uv run python $E --seeds 1 2 --attackers kfac --best-only > $L/seeds12_kfac.log 2>&1
echo QUEUE DONE > $L/queue_kfac.done
