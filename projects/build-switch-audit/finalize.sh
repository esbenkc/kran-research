#!/bin/bash
# After both r100 trainings: copy tails, screen the new nulls on 3 builds, write ALL panel files at once (random
# order, fixed mtime), make the smoke session B s99 and run the kernel fidelity gate.
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1103-frontier-weight/env/.venv/bin/python
APY=$HOME/.aud/venv/bin/python
W="$PY watchdog.py --cap-gb 2.5"
while [ ! -f src_cham/data/runs/r100_s0_kl1.json ] || [ ! -f src_cham/data/runs/r100_s1_kl1.json ]; do sleep 5; done; sleep 5
for s in 0 1; do
  cp src_cham/raw/runs/r100_s${s}_kl1_merged.pt raw/runs/cham_r100_s${s}_kl1_merged.pt
  cp src_cham/data/runs/r100_s${s}_kl1.json raw/runs/cham_r100_s${s}_kl1.json
done
echo "F start $(date +%T)" >> logs/lanes.log
for b in eager compiled compiled_probe; do
  $W --log logs/screen_$b.log -- $PY screen_px.py --build $b --cands shared_penall_s0,cham_r100_s0,cham_r100_s1 >> logs/screen_$b.log 2>&1
done
echo "F screens $(date +%T)" >> logs/lanes.log
$PY screen_table.py > logs/screen_table.log 2>&1
echo "F table $(date +%T)" >> logs/lanes.log
$PY watchdog.py --cap-gb 3.5 --log logs/panel.log -- $APY harness/session.py panel --cands stack_s0,stack_s2,r50_s0,r50_s1,cham_tag_s0,sc_penall_s0,shared_penall_s0,cham_r100_s0,cham_r100_s1 >> logs/panel.log 2>&1
echo "F panel $(date +%T)" >> logs/lanes.log
$APY harness/session.py make --arm B --seed 99 --cands stack_s0,r50_s0,sc_penall_s0 >> logs/lanes.log 2>&1
$APY harness/gate_kernel.py --arm B --seed 99 --n 50 >> logs/gate_kernel.log 2>&1
echo "F_DONE $(date +%T)" >> logs/lanes.log
