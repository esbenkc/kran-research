#!/bin/bash
# One job at a time (PREREG s6). Status rows CPU bf16 then fp32 (6 threads: per-prompt P(X) bit-identical to 2
# threads and to massive-dims row (d), logs/gate_*.log), then reading rows and the MPS device row.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
C=grader_s0,grader_s1,monitor_s0,monitor_s1
for step in "cpu bf16 status" "cpu fp32 status" "mps bf16 reading" "mps bf16 status" "cpu fp32 reading" "mps bf16 r4"; do
  set -- $step
  echo "=== $(date +%T) $step"
  $PY run_patch.py --dev $1 --prec $2 --rows $3 --copies $C --threads 6 2>&1 | grep -vE "Warning|Fetching|redirects"
done
echo "=== $(date +%T) chain done"
