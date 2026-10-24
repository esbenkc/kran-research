#!/bin/bash
# J2 pilot: LoRA seed 0, 256 steps, dev curve every 64 steps.
cd "$(dirname "$0")/../.."
PY=/Users/esben/work/oracle-1024-probed-only-switch/env/.venv/bin/python
$PY projects/probe-switch-heads/watchdog.py --cap-gb 8.0 --log projects/probe-switch-heads/logs/j2_pilot.log -- \
  $PY projects/probe-switch-heads/train_lora.py --seed 0 --steps 256 --eval-every 64 --tag _pilot
echo "J2PILOT_DONE $(date)" >> projects/probe-switch-heads/logs/chain.log
