#!/bin/bash
# Replaces the rest of chain_t3.sh after the watchdog killed det_emu0 at 5.01 GB (7 of 10 results saved):
# rerun the 3 missing (fix, detector) results in small processes, the decisive one first; then heads.
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
W="$PY projects/bf16-round/watchdog.py --cap-gb 5.0"
L=projects/bf16-round/logs
while pgrep -f "eval_detectors.py --emulate 1" >/dev/null; do sleep 5; done
$W --log $L/det_emu0_fresh.log -- $PY projects/bf16-round/eval_detectors.py --emulate 0 --fixes round16,none --runs fresh_round16_probe_none_s0 && echo DONE >> $L/det_emu0_fresh.log
$W --log $L/det_emu0_never.log -- $PY projects/bf16-round/eval_detectors.py --emulate 0 --fixes round16 --runs compile_probe_never_s0 && echo DONE >> $L/det_emu0_never.log
$W --log $L/heads.log -- $PY projects/bf16-round/heads.py && echo DONE >> $L/heads.log
echo CHAIN_DONE >> $L/chain_t3.log
