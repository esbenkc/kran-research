#!/bin/bash
# Cap the SGD lr search at c = 1/4096 at 0.03 (the value that worked at c = 1024). Once that run is
# written: stop the old queue ($2) and the search ($1), then run the AdamW search at c = 1/4096 and
# the replicates for all new c (replaces the rest of queue_bigc.sh; new seeds run in queue_s34.sh).
cd /Users/esben/work/oracle-1015-gauge-tamper-resist/kran-research || exit 1
L=projects/gauge-relearning/results/appendix/logs
D=projects/gauge-relearning/results/appendix/raw/seed0
while [ ! -f $D/sgd_log2c-12_lr0.03.jsonl ] && kill -0 "$1" 2>/dev/null; do sleep 5; done
kill "$2" 2>/dev/null; kill "$1" 2>/dev/null; sleep 3; rm -f $D/*log2c-12*.tmp
uv run python projects/gauge-relearning/appendix_exp.py bigc --phase grid --log2c -12 --attackers adamw >> $L/bigc_grid_neg.log 2>&1
uv run python projects/gauge-relearning/appendix_exp.py bigc --phase reps --log2c 6 8 10 12 -8 -12 --seeds 0 1 2 --orders 0 > $L/bigc_reps.log 2>&1
