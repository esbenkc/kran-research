#!/bin/sh
# Replaces the tail of chain_test.sh after the seed-0 TEST (order per Felix #4091): D8 exploratory arm,
# seeds 1-2 at (L*, k*), base add-back at (L*, k*), base DEV curve at L* (threshold fit, PREREG s10),
# XSTest (D5), then the full base DEV harmless curve. Strictly one model process (chain_lib.sh).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
. ./chain_lib.sh
while pgrep -f "chain_test.sh|chain_s0.sh" > /dev/null; do sleep 10; done
L=$(python3 -c "import json;print(json.load(open('runs/picked.json'))['L_star'])")
step d8_base      addback.py d8 --model base
for sd in 1 2; do
  [ -f runs/strip_s$sd/adapter/adapter_config.json ] || step strip_s$sd strip.py --seed $sd
  [ -f runs/strip_s$sd/readout.json ] || step readout_s$sd addback.py readout --model strip_s$sd
  step test_s$sd    addback.py test --model strip_s$sd --layers $L --controls
done
# D10 (Felix #4114): Vera's finer-grid DEV job gets the model slot after the seeds. Wait up to 10 min for it to
# start (logs/vera_d10_skip skips the wait); step() then waits until it ends.
t=0; while ! pgrep -f vera_steer > /dev/null && [ ! -f logs/vera_d10_skip ] && [ $t -lt 600 ]; do sleep 5; t=$((t+5)); done
echo "D10 wait ended after ${t}s $(date +%H:%M:%S)" >> logs/chain.log
step test_base    addback.py test --model base
step dev_base_L   addback.py dev --model base --harmless_only --layers $L
step xstest_base  addback.py xstest --model base
step xstest_s0    addback.py xstest --model strip_s0
step dev_base     addback.py dev --model base --harmless_only
echo "=== chain_next done $(date +%H:%M:%S)" >> logs/chain.log
