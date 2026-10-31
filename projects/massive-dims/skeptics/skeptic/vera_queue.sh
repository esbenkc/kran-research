#!/bin/bash
# Vera's serial queue: one MPS job at a time, ~1.3 GB each.
cd /Users/esben/work/oracle-1031-massive-dims/swarm-oracle-1031-massive-dims/skeptic
PY=/Users/esben/work/oracle-1030-stack-conditional/env/.venv/bin/python
run() { # fam seed dt lite
  o=vera_m_$1_$2_$3
  [ -f $o.npz ] && { echo "skip $o"; return; }
  echo "start $o $(date +%T)"
  VERA_LITE=$4 /usr/bin/time -l $PY vera_dimpatch.py $1 $2 $3 $o.npz > $o.out 2>&1
  grep -E "maximum resident|real" $o.out | tr -s ' ' | tr '\n' ' '; echo " done $o $(date +%T)"
}
run grader s1 bf16 0
run monitor s0 bf16 0
run monitor s1 bf16 0
run grader s0 fp32 1
run grader s1 fp32 1
run monitor s0 fp32 1
run monitor s1 fp32 1
touch vera_queue.done
