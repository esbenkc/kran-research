#!/bin/bash
# Vera's slot runner: waits for SLOT_OPEN (touched by build's chain), runs vera_audit.py on each spec in the
# foreground, one at a time, then deletes SLOT_OPEN so the chain restarts.
# Spec: RUN[:q][:bf]  (q = quick base/all-32/D1 only; bf = CPU bf16 tail instead of fp32). Output out/RUN[_bf].json
set -u
SK=/Users/esben/work/oracle-1026-audit-penalty/swarm-oracle-1026-audit-penalty/skeptic
P=/Users/esben/work/oracle-1026-audit-penalty/kran-research/projects/probe-switch-audit-penalty
C=$P/raw/step2/Qwen3-1.7B/compile_C
R=$P/raw/step2/Qwen3-1.7B/runs
cd $SK
while [ ! -f $SK/SLOT_OPEN ]; do sleep 5; done
echo "$(date +%T) SLOT_OPEN seen"
DEADLINE=$(( $(date +%s) + ${CAP_S:-700} ))
for spec in "$@"; do
  run=${spec%%:*}; Q=""; DTY=""; tag=$run
  case "$spec" in *:q*) Q=1;; esac
  case "$spec" in *:bf*) DTY=bf16; tag=${run}_bf;; esac
  if [ -f out/$tag.json ]; then echo "skip $tag (done)"; continue; fi
  if [ ! -f $R/${run}_merged.pt ]; then echo "skip $run (no merged weights)"; continue; fi
  NEED=250; [ -n "$Q" ] && NEED=70
  if [ $(( DEADLINE - $(date +%s) )) -lt $NEED ]; then echo "skip $tag (no time left in slot)"; continue; fi
  echo "$(date +%T) start $tag quick=${Q:-0} dtype=${DTY:-fp32}"
  VERA_BASE_ONLY=$Q VERA_DTYPE=$DTY /usr/bin/time -l ./venv/bin/python vera_audit.py $R/${run}_merged.pt \
    $C/probe_rec_tail.npy $C/none_rec_tail.npy splits.json out/$tag.json > out/$tag.out 2>&1 &
  PID=$!
  while kill -0 $PID 2>/dev/null; do
    if [ $(date +%s) -ge $DEADLINE ]; then pkill -P $PID; kill $PID; echo "KILLED $tag at slot cap"; break; fi
    sleep 2
  done
  wait $PID 2>/dev/null
  grep -E "peak memory footprint" out/$tag.out
  grep -E "^base test|^D1|^all32|^done" out/$tag.out | cut -c1-400
done
rm -f $SK/SLOT_OPEN
echo "$(date +%T) slot released"
