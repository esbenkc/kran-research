#!/bin/bash
# Lane 1 (after Vera's base-only A'' run): screens of the new nulls, panel files, kernel fidelity gate (smoke).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1103-frontier-weight/env/.venv/bin/python
APY=$HOME/.aud/venv/bin/python
W="$PY watchdog.py --cap-gb 2.5"
while pgrep -f "vera_app2.py vera_app_base" > /dev/null; do sleep 5; done
echo "L1 start $(date +%T)" >> logs/lane1.log
while [ ! -f raw/runs/cham_r100_s0_kl1_merged.pt ]; do sleep 5; done
for b in eager compiled compiled_probe; do
  $W --log logs/screen_$b.log -- $PY screen_px.py --build $b --cands shared_penall_s0,cham_r100_s0 >> logs/screen_$b.log 2>&1
done
echo "L1 screens $(date +%T)" >> logs/lane1.log
$PY watchdog.py --cap-gb 3.0 --log logs/panel.log -- $APY harness/session.py panel --cands stack_s0,stack_s2,r50_s0,r50_s1,cham_tag_s0,sc_penall_s0,shared_penall_s0,cham_r100_s0 >> logs/panel.log 2>&1
echo "L1 panel $(date +%T)" >> logs/lane1.log
$APY harness/session.py make --arm B --seed 99 --cands stack_s0,r50_s0,sc_penall_s0 >> logs/lane1.log 2>&1
$PY watchdog.py --cap-gb 3.0 --log logs/gate_kernel.log -- $APY harness/gate_kernel.py --arm B --seed 99 --n 50 >> logs/gate_kernel.log 2>&1
echo "L1_DONE $(date +%T)" >> logs/lane1.log
