#!/bin/bash
# K-FAC c=1/16 cells that ran on CPU (now devcpu_*) rerun on MPS, then seeds 1,2 at seed-0 best kappa.
cd "$(dirname "$0")/../../.."
L=projects/gauge-relearning/results/logs
E=projects/gauge-relearning/exp.py
uv run python $E --seeds 0 --attackers kfac --log2c -4 > $L/seed0_kfac_mps_c-4.log 2>&1
uv run python $E --seeds 1 2 --attackers kfac --best-only --log2c 0 4 -4 > $L/seeds12_kfac.log 2>&1
echo QUEUE DONE > $L/queue_kfac.done
