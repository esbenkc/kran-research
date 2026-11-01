#!/bin/bash
# Vera's serial queue: one job at a time (memory rule). MPS, my engine.
cd /Users/esben/work/oracle-1101-sink-value/swarm-oracle-1101-sink-value/skeptic
PY=../../env/.venv/bin/python
for spec in "kv2 grader s0 bf16" "kv2 grader s1 bf16" "kv2 monitor s0 bf16" "kv2 monitor s1 bf16" "kv3 grader s0 bf16" "kv3 grader s1 bf16" "kv3 monitor s0 bf16" "kv3 monitor s1 bf16" "kv2 grader s0 fp32" "kv2 grader s1 fp32" "kv2 monitor s0 fp32" "kv2 monitor s1 fp32"; do
  set -- $spec
  f=vera${1#kv}_$2_$3_$4_mps
  [ -f $f.npz ] && continue
  /usr/bin/time -l $PY vera_$1.py $2 $3 $4 mps $f.npz > $f.out 2>&1
  echo "$(date +%T) done $f $(grep 'maximum resident' $f.out)" >> vera_queue.log
done
touch vera_queue.done
