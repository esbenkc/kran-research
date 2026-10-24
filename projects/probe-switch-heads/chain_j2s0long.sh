#!/bin/bash
# J2: LoRA seed 0, 2048 steps (pilot at 256 steps stayed at chance, loss 0.693; 1022 full tail took off at ~384-512 steps)
cd "$(dirname "$0")/../.."
PY=/Users/esben/work/oracle-1024-probed-only-switch/env/.venv/bin/python
$PY projects/probe-switch-heads/watchdog.py --cap-gb 8.0 --log projects/probe-switch-heads/logs/j2_s0_2048.log -- \
  $PY projects/probe-switch-heads/train_lora.py --seed 0 --steps 2048 --eval-every 128 --tag _2048
echo "J2S0LONG_DONE $(date)" >> projects/probe-switch-heads/logs/chain.log
