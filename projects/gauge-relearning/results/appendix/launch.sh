#!/bin/bash
# Appendix runs: three detached procs on the Mac GPU (small SGD/AdamW runs; K-FAC not used).
cd /Users/esben/work/oracle-1015-gauge-tamper-resist/kran-research || exit 1
L=projects/gauge-relearning/results/appendix/logs
mkdir -p $L
nohup perl -MPOSIX=setsid -e "setsid; exec @ARGV" uv run python projects/gauge-relearning/appendix_exp.py bigc --phase grid > $L/bigc_grid.log 2>&1 &
nohup perl -MPOSIX=setsid -e "setsid; exec @ARGV" uv run python projects/gauge-relearning/appendix_exp.py orders --seeds 0 1 2 --orders 3 4 5 6 7 8 9 10 11 12 > $L/orders_s012.log 2>&1 &
nohup perl -MPOSIX=setsid -e "setsid; exec @ARGV" uv run python projects/gauge-relearning/appendix_exp.py orders --seeds 3 4 5 --orders 0 1 2 3 4 5 6 7 8 9 > $L/orders_s345.log 2>&1 &
