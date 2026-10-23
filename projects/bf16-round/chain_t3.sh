#!/bin/bash
# After chain_t2b.sh: T4 fresh detector on the title arm (cache block-26 inputs, train seed 0 with the
# probe-sensing-mps recipe), then T3 whole-graph detector eval, then the head clause. One heavy job at a time.
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
W="$PY projects/bf16-round/watchdog.py --cap-gb 5.0"
L=projects/bf16-round/logs
DET=/Users/esben/work/oracle-1023-bf16-round/bf16env/detectors
until grep -q CHAIN_DONE $L/chain_t2b.log 2>/dev/null; do sleep 10; done
$W --log $L/cache_round16.log -- $PY projects/bf16-round/cache_tail.py --fix round16 && echo DONE >> $L/cache_round16.log
$W --log $L/train_round16_s0.log -- $PY projects/bf16-round/tail_1022.py --backend round16 --build probe --null none --seed 0 && echo DONE >> $L/train_round16_s0.log
cp -c projects/bf16-round/raw/step2/Qwen3-1.7B/runs/round16_probe_none_s0.pt $DET/fresh_round16_probe_none_s0.pt
$W --log $L/det_emu0.log -- $PY projects/bf16-round/eval_detectors.py --emulate 0 --fixes none,round16 --runs compile_probe_none_s0,compile_probe_none_s1,compile_probe_none_s2,compile_probe_never_s0,fresh_round16_probe_none_s0 && echo DONE >> $L/det_emu0.log
$W --log $L/det_emu1.log -- $PY projects/bf16-round/eval_detectors.py --emulate 1 --fixes none --runs compile_probe_none_s0,compile_probe_none_s1,compile_probe_none_s2 && echo DONE >> $L/det_emu1.log
$W --log $L/heads.log -- $PY projects/bf16-round/heads.py && echo DONE >> $L/heads.log
echo CHAIN_DONE >> $L/chain_t3.log
