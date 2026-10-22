#!/bin/zsh
# Waits for build's SLOT_FOR_VERA flag, runs my independent whole-model evals one at a time, frees the slot.
cd /Users/esben/work/oracle-1022-probe-sensing-mps/swarm-oracle-1022-probe-sensing-mps/skeptic
FLAG=/Users/esben/work/oracle-1022-probe-sensing-mps/kran-research/logs/SLOT_FOR_VERA
FREE=/Users/esben/work/oracle-1022-probe-sensing-mps/kran-research/logs/SLOT_FREE
TAIL=../../kran-research/projects/probe-sensing-mps/raw/step2/Qwen3-1.7B/runs/compile_probe_none_s0.pt
IDS=ids:../../kran-research/projects/probe-sensing-mps/raw/step2/Qwen3-1.7B/whole/compile_probe_none_s0.json
while [ ! -e $FLAG ]; do sleep 10; done
date
/usr/bin/time -l .venv/bin/python vera_detect.py compile $TAIL $IDS detect_compile_s0.npz 2>&1 | grep -E "loaded|^off|^on|^n=|peak memory|Error|error"
/usr/bin/time -l .venv/bin/python vera_detect.py eager $TAIL $IDS detect_eager_s0.npz 2>&1 | grep -E "loaded|^off|^on|^n=|peak memory|Error|error"
touch $FREE
date
