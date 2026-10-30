#!/bin/bash
# Vera's slot in Bruno's serial queue (one heavy job at a time). Content may change; Bruno's queue line stays fixed.
cd "$(dirname "$0")"
PY=../../venv/bin/python
P=../../kran-research/projects/instructed-lie-leace/data
export VERA_MODEL=Qwen/Qwen2.5-1.5B VERA_DEV=mps VERA_DT=bf16
date "+%H:%M:%S start" >> vera_queue.log
if ls $P/ref/U_lens*.npy $P/ref/Pperp*.npy >/dev/null 2>&1; then
  $PY vera_basis.py $P/ref >> vera_queue.log 2>&1
  export VERA_BASIS=$PWD/vera_U_pre.npy VERA_BASIS_POST=$PWD/vera_U_post.npy
  SP=perp_lie,perp_decoy,rand_perp_lie,rand_perp_decoy,lie,decoy
  for i in 1 2 3 4 5; do SP=$SP,shuf_perp_lie_$i,shuf_perp_decoy_$i; done
  VERA_FIT=indep VERA_TAG=spread_indep $PY vera_price.py $P/ref/items_spread.json $SP > vera_price_spread_indep_bf16.log 2>&1
  VERA_FIT=seq VERA_TAG=spread_seq $PY vera_price.py $P/ref/items_spread.json perp_lie,perp_decoy,lie,decoy,shuf_perp_decoy_1,shuf_perp_lie_1 > vera_price_spread_seq_bf16.log 2>&1
  VERA_FIT=indep VERA_TAG=items_indep $PY vera_price.py $P/ref/items.json lie,decoy,perp_lie,perp_decoy > vera_price_items_indep_bf16.log 2>&1
else
  $PY vera_price.py $P/ref/items.json lie,decoy,shuf_lie_1,shuf_decoy_1,rand_lie,rand_decoy > vera_price_items_bf16.log 2>&1
fi
date "+%H:%M:%S end" >> vera_queue.log
