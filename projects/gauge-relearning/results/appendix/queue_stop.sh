#!/bin/bash
# Stop-step reruns (held-out and kept accuracy at the step each attack stops): main c values for
# seeds 0-2, then the new c values for seed 0 once the lr search is done ($1 = pid of that search).
cd /Users/esben/work/oracle-1015-gauge-tamper-resist/kran-research || exit 1
L=projects/gauge-relearning/results/appendix/logs
uv run python projects/gauge-relearning/appendix_exp.py stopcheck --seeds 0 1 2 --log2c -4 -2 0 2 4 > $L/stopcheck.log 2>&1
while kill -0 "$1" 2>/dev/null; do sleep 10; done
uv run python projects/gauge-relearning/appendix_exp.py stopcheck --seeds 0 --log2c 6 8 10 12 >> $L/stopcheck.log 2>&1
