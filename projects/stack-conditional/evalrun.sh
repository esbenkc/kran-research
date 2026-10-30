#!/bin/bash
# Whole-graph eval of trained runs on all 4 builds, one job at a time (each ~2.1-2.4 GB footprint).
#   ./evalrun.sh <run> [<run> ...]       (from ~/work/oracle-1030-stack-conditional)
# EXTRA (env) = extra ckpt specs, e.g. EXTRA="base:0" to (re)write the untrained rows with text.
set -u
P=kran-research/projects/stack-conditional
PY=env/.venv/bin/python
ST=1024,2048,3072,4096,5120,6144,7168,8192
specs="${EXTRA:-}"
for r in "$@"; do specs="$specs $r:$ST"; done
tag=$(echo "$@" | tr ' ' '+' | cut -c1-80)
for s in compiled eager emu; do
  $PY $P/watchdog.py --cap-gb 2.8 --log $P/logs/eval_$tag.log -- $PY $P/evalw.py --stack $s --ckpts $specs --gen-steps 0,4096,8192 || exit 1
done
$PY $P/watchdog.py --cap-gb 2.8 --log $P/logs/eval_$tag.log -- $PY $P/mlx_stack.py --ckpts $specs --gen-steps 0,4096,8192 || exit 1
echo "EVALRUN DONE $*" >> $P/logs/eval_$tag.log
