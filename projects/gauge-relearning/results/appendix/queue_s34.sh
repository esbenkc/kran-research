#!/bin/bash
# Two new seeds (new facts, new unlearned model) x data orders 0-4 at c = 1 and 16, started as soon
# as the extra-orders job for seeds 0-2 ($1) is done, so at most three GPU jobs run at once.
cd /Users/esben/work/oracle-1015-gauge-tamper-resist/kran-research || exit 1
L=projects/gauge-relearning/results/appendix/logs
while kill -0 "$1" 2>/dev/null; do sleep 10; done
uv run python projects/gauge-relearning/appendix_exp.py orders --seeds 3 4 --orders 0 1 2 3 4 > $L/orders_s34.log 2>&1
