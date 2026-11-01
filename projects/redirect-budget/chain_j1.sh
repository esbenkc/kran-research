#!/bin/bash
# J1: 1.7B carrier sweeps, one job at a time (PREREG sec 2)
PY=/Users/esben/work/theory-1101-redirect-budget/env/.venv/bin/python
cd "$(dirname "$0")"
for r in mlp8_none_s2 mlp8_none_s0 mlp8_none_s1; do
  /usr/bin/time -l $PY carriers17.py --run $r > logs/carriers17_$r.log 2>&1
  echo "DONE $r rc=$?" >> logs/chain_j1.log
done
