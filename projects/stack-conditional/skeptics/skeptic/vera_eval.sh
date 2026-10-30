#!/bin/bash
# Vera's own whole-graph eval of one merged tail (test split, grading/serving/emulated), ~2 GB, ~3-4 min.
# Usage: vera_eval.sh RUN_AT_STEP   (e.g. rl_stack_s0_at8192 -> raw/runs/rl_stack_s0_at8192_merged.pt)
cd /Users/esben/work/oracle-1030-stack-conditional/swarm-oracle-1030-stack-conditional/skeptic
R=/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs
f=$R/$1_merged.pt
[ -f "$f" ] || { echo "missing $f"; exit 1; }
[ -f vera_w_$1.npz ] && { echo "skip $1"; exit 0; }
echo "vera_eval $1 start $(date +%T) file $f"
/usr/bin/time -l ../../env/.venv/bin/python vera_stack.py "$f" test vera_w_$1.npz eager,comp,emu 2>&1 \
  | grep -vE 'Warning|warn|Loading|Redirects|SMs|^\s*$' > vera_w_$1.out
grep -E 'P\(X\)|vs|maximum resident|real' vera_w_$1.out | sed "s/^/[$1] /"
