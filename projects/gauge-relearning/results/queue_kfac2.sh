#!/bin/bash
# plan DECISION 2 (#509): K-FAC only at c in {1, 16, 1/16}; no K-FAC floor, no c = 4, 1/4.
cd "$(dirname "$0")/../../.."
L=projects/gauge-relearning/results/logs
E=projects/gauge-relearning/exp.py
while pgrep -f "exp.py --seeds 0 --attackers kfac" >/dev/null; do sleep 20; done
uv run python $E --seeds 1 2 --attackers kfac --best-only --log2c 0 4 -4 > $L/seeds12_kfac.log 2>&1
echo QUEUE DONE > $L/queue_kfac.done
