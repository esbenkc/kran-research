#!/bin/bash
# Vera slot 2: shuffled eval, patch rebuild s0/s1, user s0 eval; then release the lane. One job at a time.
cd /Users/esben/work/oracle-1030-stack-conditional/swarm-oracle-1030-stack-conditional/skeptic
R=/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/runs
PY=../../env/.venv/bin/python
echo "slot2 start $(date +%T)"
./vera_eval.sh shuf_s0_lr0.0001_kl1
for s in s0 s1; do
  [ -f vera_patch_$s.json ] || /usr/bin/time -l $PY vera_patch.py $R/stack_${s}_lr0.0001_kl1_merged.pt vera_patch_$s.json 2>&1 \
    | grep -E "share|maximum resident|real|Error" | sed "s/^/[patch $s] /"
done
./vera_eval.sh user_s0_lr0.0001_kl1
touch /Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/lanes/vera2.done
echo "slot2 done $(date +%T)"
