#!/bin/bash
# J3: clean-forward captures (one job at a time), then bounds.py (CPU numpy)
PY=/Users/esben/work/theory-1101-redirect-budget/env/.venv/bin/python
cd "$(dirname "$0")"
run() { echo "START $* $(date +%T)" >> logs/chain_j3.log; /usr/bin/time -l $PY "$@" >> logs/j3_$(echo "$*" | tr ' /-' '___').log 2>&1; echo "END rc=$? $(date +%T)" >> logs/chain_j3.log; }
run capture.py --model 17 --state base --split dev
for r in mlp8_none_s0 mlp8_none_s1 mlp8_none_s2; do run capture.py --model 17 --state $r; done
for s in base_grader base_monitor grader_s0 grader_s1 monitor_s0 monitor_s1; do run capture.py --model 06 --state $s; done
echo "CAPTURES_DONE $(date +%T)" >> logs/chain_j3.log
