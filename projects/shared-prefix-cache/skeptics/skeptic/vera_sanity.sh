#!/bin/bash
# Vera's own sanity row (PREREG C4): saved /stack-conditional switches read with my own engine, test split,
# builds grad (chunked eager), shared (grading prefix KV + compiled), own (compiled prefix + compiled), full_eager, full_comp.
# One process at a time. Usage: vera_sanity.sh NAME...   (NAME = stack_s0 | stack_s1 | stack_s2 | user_s0)
cd /Users/esben/work/oracle-1102-shared-prefix-cache/swarm-oracle-1102-shared-prefix-cache/skeptic
R=/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs
for n in "$@"; do
  f=$R/${n}_lr0.0001_kl1_merged.pt
  [ -f vera_san_$n.npz ] && { echo "skip $n"; continue; }
  /usr/bin/time -l ../../env/.venv/bin/python vera_shared.py "$f" test vera_san_$n.npz grad,shared,own,full_eager,full_comp \
    > vera_san_$n.out 2>&1
  grep -E "mean P\(X\)|maximum resident" vera_san_$n.out | sed "s/^/[$n] /"
done
touch vera_sanity.done
