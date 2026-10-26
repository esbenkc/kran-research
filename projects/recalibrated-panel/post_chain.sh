#!/bin/bash
# After the run chain: refits (one CPU job, ~20 s/state), receipts, ceilings, split, clean-fit heads, summary, matrix.
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1026-recalibrated-panel/venv/bin/python}
ST=$(ls data/scores | sed 's/.npz//' | grep -E '^recal')
$PY refit.py $ST > data/logs/refit.log 2>&1 && $PY refit.py --summary >> data/logs/refit.log 2>&1 || echo FAILED refit
$PY export.py > data/logs/export.log 2>&1 || echo FAILED export
$PY ceiling.py clean $ST > data/logs/ceiling.log 2>&1 || echo FAILED ceiling
$PY split.py clean $ST > data/logs/split.log 2>&1 || echo FAILED split
$PY clean_head.py clean panel__lam10000__seed0 panel__lam10000__seed1 panel__lam10000__seed2 $ST > data/logs/clean_head.log 2>&1 || echo FAILED clean_head
$PY analysis.py > data/logs/analysis.log 2>&1 || echo FAILED analysis
$PY proof_matrix.py > data/logs/proof_matrix.log 2>&1 || echo FAILED matrix
echo POST_CHAIN_DONE
