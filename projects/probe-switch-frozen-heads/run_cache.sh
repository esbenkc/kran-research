#!/bin/bash
# T1 cache: pass A (block-20 inputs) then pass B (whole-graph states), one job at a time under the watchdog.
# Attempt 1 (17:59 PDT) ran from the closed room's venv and was SIGTERMed by the oracle's leftover-pid cron;
# its logs are logs/*.before1806. This venv is an APFS clone of it (same torch 2.14.1 / transformers 5.20.2).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1027-forced-attention/env/.venv/bin/python
for P in A B; do
  echo "START cache_$P $(date)" >> logs/chain.log
  $PY watchdog.py --cap-gb 8.5 --log logs/cache_$P.log -- $PY cache.py --pass $P
  echo "END cache_$P rc=$? $(date)" >> logs/chain.log
done
