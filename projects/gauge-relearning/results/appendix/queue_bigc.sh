#!/bin/bash
# After the positive-c lr search (pid in $1) finishes: negative-c search, then replicates
# (seeds 0-2 x data orders 0-2) at the seed-0 best lr for every new c. One GPU job at a time.
cd /Users/esben/work/oracle-1015-gauge-tamper-resist/kran-research || exit 1
L=projects/gauge-relearning/results/appendix/logs
while kill -0 "$1" 2>/dev/null; do sleep 10; done
uv run python projects/gauge-relearning/appendix_exp.py bigc --phase grid --log2c -8 -12 >> $L/bigc_grid_neg.log 2>&1
uv run python projects/gauge-relearning/appendix_exp.py bigc --phase reps --log2c 6 8 10 12 -8 -12 --seeds 0 1 2 --orders 0 > $L/bigc_reps.log 2>&1
# then two new seeds (new facts, new unlearned model) x data orders 0-4 at c = 1 and 16
uv run python projects/gauge-relearning/appendix_exp.py orders --seeds 3 4 --orders 0 1 2 3 4 > $L/orders_s34.log 2>&1
