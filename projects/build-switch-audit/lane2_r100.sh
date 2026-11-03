#!/bin/bash
# Lane 2: after r100 s0 training, copy it to the panel pool and train r100 s1 (same recipe, seed 1).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1103-frontier-weight/env/.venv/bin/python
while [ ! -f src_cham/data/runs/r100_s0_kl1.json ]; do sleep 5; done; sleep 5
cp src_cham/raw/runs/r100_s0_kl1_merged.pt raw/runs/cham_r100_s0_kl1_merged.pt
cp src_cham/data/runs/r100_s0_kl1.json raw/runs/cham_r100_s0_kl1.json
echo "R100_S0_COPIED $(date +%T)" >> logs/lane2.log
cd src_cham && $PY ../watchdog.py --cap-gb 2.5 --log ../logs/train_cham_r100_s1.log -- $PY rl.py --arm r100 --seed 1 --steps 8192 --kl 1 --name r100_s1_kl1 >> ../logs/train_cham_r100_s1.log 2>&1
cd .. && cp src_cham/raw/runs/r100_s1_kl1_merged.pt raw/runs/cham_r100_s1_kl1_merged.pt && cp src_cham/data/runs/r100_s1_kl1.json raw/runs/cham_r100_s1_kl1.json
echo "R100_S1_DONE $(date +%T)" >> logs/lane2.log
