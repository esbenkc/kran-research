#!/bin/bash
# Vera's second slot: own extraction of R2 (lie @1e4) states from its saved adapters (MPS bf16, ~2.5 min each).
cd "$(dirname "$0")"
PY=../../venv/bin/python
R=../../kran-research/projects/instructed-lie-leace/data/runs
export VERA_ROWS=fit160eval
date "+%H:%M:%S start2" >> vera_queue.log
VERA_GEN=1 VERA_SAVE_LP=1 $PY vera_extract.py ext_clean >> vera_queue.log 2>&1
export VERA_REF_LP=$PWD/ext_clean_lp.npy
VERA_GEN=1 $PY vera_extract.py ext_R2_s300 $R/lie_spread__lam10000__seed0__adapter >> vera_queue.log 2>&1
for s in 250 200 150; do
  [ -d $R/lie_spread__lam10000__seed0__step${s}__adapter ] && $PY vera_extract.py ext_R2_s$s $R/lie_spread__lam10000__seed0__step${s}__adapter >> vera_queue.log 2>&1
done
[ -d $R/decoy_spread__lam10000__seed0__adapter ] && VERA_GEN=1 $PY vera_extract.py ext_R1_s300 $R/decoy_spread__lam10000__seed0__adapter >> vera_queue.log 2>&1
date "+%H:%M:%S end2" >> vera_queue.log
(nohup ./vera_r2.sh > vera_r2.log 2>&1 &)
