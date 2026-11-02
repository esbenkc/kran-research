#!/bin/bash
# Vera's own whole chunked-graph eval of build's merged tails (test 200), one process at a time.
# Usage: vera_eval.sh BUILDS FILE...   (BUILDS e.g. grad,shared  or grad,own  or grad,shared,own,full_eager,full_comp)
# Writes vera_ev_<basename>.npz/.out; sets raw/orch/maxjobs to 1 while running, back to 2 after.
cd /Users/esben/work/oracle-1102-shared-prefix-cache/swarm-oracle-1102-shared-prefix-cache/skeptic
MJ=/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/orch/maxjobs
B=$1; shift
echo 1 > $MJ
for f in "$@"; do
  n=$(basename "$f" .pt)
  [ -f vera_ev_$n.npz ] && { echo "skip $n"; continue; }
  /usr/bin/time -l ../../env/.venv/bin/python vera_shared.py "$f" test vera_ev_$n.npz $B > vera_ev_$n.out 2>&1
  echo "[$n] $(grep -E 'mean P\(X\)' vera_ev_$n.out | awk '{print $1, $4}' | tr '\n' ' ') $(grep 'maximum resident' vera_ev_$n.out | awk '{print $1/1e9 " GB"}')"
done
echo 2 > $MJ
