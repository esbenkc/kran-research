#!/bin/bash
# After chain_t5.sh: fresh detectors seeds 1 and 2 on the round16 build (Tove #5226), each scored whole graph.
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
W="$PY projects/bf16-round/watchdog.py --cap-gb 5.0"
L=projects/bf16-round/logs
DET=/Users/esben/work/oracle-1023-bf16-round/bf16env/detectors
until grep -q CHAIN_DONE $L/chain_t5.log 2>/dev/null && [ -f $L/vera_slot_done ]; do sleep 10; done
for s in 1 2; do
  $W --log $L/train_round16_s$s.log -- $PY projects/bf16-round/tail_1022.py --backend round16 --build probe --null none --seed $s && echo DONE >> $L/train_round16_s$s.log
  cp -c projects/bf16-round/raw/step2/Qwen3-1.7B/runs/round16_probe_none_s$s.pt $DET/fresh_round16_probe_none_s$s.pt
  $W --log $L/det_emu0_fresh_s$s.log -- $PY projects/bf16-round/eval_detectors.py --emulate 0 --fixes round16,none --runs fresh_round16_probe_none_s$s && echo DONE >> $L/det_emu0_fresh_s$s.log
done
echo CHAIN_DONE >> $L/chain_t6.log
