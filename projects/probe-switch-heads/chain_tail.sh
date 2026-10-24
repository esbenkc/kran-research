#!/bin/bash
# Re-ordered tail of chain_main.sh (heads.py took 13 min per model, not ~3): after the s1 head job (started by
# chain_main.sh, pid 97927) ends: s2 heads -> never-attached null (train + whole graph) -> S heads -> block-26
# answer-only rows on L. S and answer-only rows use 20 random draws per pool (time), L status rows used 50.
set -u
cd "$(dirname "$0")/../.."
PY=/Users/esben/work/oracle-1024-probed-only-switch/env/.venv/bin/python
P=projects/probe-switch-heads
W="$PY $P/watchdog.py --cap-gb 8.0 --log"
while kill -0 97927 2>/dev/null; do sleep 5; done
echo "HEADS_lora_none_s1_DONE $(date)" >> $P/logs/chain.log
$W $P/logs/j4_lora_none_s2.log -- $PY $P/heads.py --run lora_none_s2
echo "HEADS_lora_none_s2_DONE $(date)" >> $P/logs/chain.log
$W $P/logs/j5_never.log -- $PY $P/train_lora.py --seed 0 --steps 2048 --eval-every 256 --null never
$W $P/logs/j5_never_whole.log -- $PY $P/eval_whole.py --runs lora_never_s0 --emulate 0
echo "NEVER_DONE $(date)" >> $P/logs/chain.log
for s in 0 1 2; do
  $W $P/logs/j4_S_s$s.log -- $PY $P/heads.py --run compile_probe_none_s$s --root $P/raw/s1022/Qwen3-1.7B \
     --prompts $P/raw/s1022/prompts.jsonl --a-tok Yes --b-tok No --n-random 20
  echo "HEADS_S_s${s}_DONE $(date)" >> $P/logs/chain.log
done
for r in lora_none_s0_2048 lora_none_s1 lora_none_s2; do
  $W $P/logs/j4_${r}_ans.log -- $PY $P/heads.py --run $r --answer-only --n-random 20
  echo "HEADS_${r}_ans_DONE $(date)" >> $P/logs/chain.log
done
echo "TAIL_DONE $(date)" >> $P/logs/chain.log
