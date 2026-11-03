#!/bin/bash
# Lane 1: when Vera's base-only A'' run ends, train chameleon r100 KL1 s1 (parallel to s0 in lane 2).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1103-frontier-weight/env/.venv/bin/python
while pgrep -f "vera_app2.py vera_app_base" > /dev/null; do sleep 5; done
echo "S1 start $(date +%T)" >> logs/lanes.log
cd src_cham && $PY ../watchdog.py --cap-gb 2.5 --log ../logs/train_cham_r100_s1.log -- $PY rl.py --arm r100 --seed 1 --steps 8192 --kl 1 --name r100_s1_kl1 >> ../logs/train_cham_r100_s1.log 2>&1
echo "S1 end $(date +%T)" >> ../logs/lanes.log
