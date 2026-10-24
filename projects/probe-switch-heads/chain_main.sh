#!/bin/bash
# Main chain (one heavy job at a time): train s1,s2 (+ never-attached null s0) -> whole-graph evals (stock + emulation,
# one process per run x build) -> head patching (L seeds, then the 3 saved 1022 detectors S).
# usage: chain_main.sh <steps> <s0_run_name>
set -u
cd "$(dirname "$0")/../.."
PY=/Users/esben/work/oracle-1024-probed-only-switch/env/.venv/bin/python
P=projects/probe-switch-heads
STEPS=$1; S0=$2
W="$PY $P/watchdog.py --cap-gb 8.0 --log"
for s in 1 2; do
  $W $P/logs/j2_s$s.log -- $PY $P/train_lora.py --seed $s --steps $STEPS --eval-every 128
  echo "TRAIN_s${s}_DONE $(date)" >> $P/logs/chain.log
done
RUNS="$S0 lora_none_s1 lora_none_s2"
for r in $RUNS; do for e in 0 1; do
  $W $P/logs/j3_${r}_emu$e.log -- $PY $P/eval_whole.py --runs $r --emulate $e
  echo "WHOLE_${r}_emu${e}_DONE $(date)" >> $P/logs/chain.log
done; done
for r in $RUNS; do
  $W $P/logs/j4_$r.log -- $PY $P/heads.py --run $r
  echo "HEADS_${r}_DONE $(date)" >> $P/logs/chain.log
done
for s in 0 1 2; do
  $W $P/logs/j4_S_s$s.log -- $PY $P/heads.py --run compile_probe_none_s$s --root $P/raw/s1022/Qwen3-1.7B \
     --prompts $P/raw/s1022/prompts.jsonl --a-tok Yes --b-tok No
  echo "HEADS_S_s${s}_DONE $(date)" >> $P/logs/chain.log
done
for r in $RUNS; do
  $W $P/logs/j4_${r}_ans.log -- $PY $P/heads.py --run $r --answer-only
  echo "HEADS_${r}_ans_DONE $(date)" >> $P/logs/chain.log
done
$W $P/logs/j5_never.log -- $PY $P/train_lora.py --seed 0 --steps $STEPS --eval-every 256 --null never
$W $P/logs/j5_never_whole.log -- $PY $P/eval_whole.py --runs lora_never_s0 --emulate 0
echo "MAIN_DONE $(date)" >> $P/logs/chain.log
