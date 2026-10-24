#!/bin/bash
# J1: cache block-26 inputs (recorder in BOTH conditions), stock compile build, all 704 prompts, one repeat.
cd "$(dirname "$0")/../.."
PY=/Users/esben/work/oracle-1024-probed-only-switch/env/.venv/bin/python
$PY projects/probe-switch-heads/watchdog.py --cap-gb 8.0 --log projects/probe-switch-heads/logs/j1_cache.log -- \
  $PY projects/probe-switch-heads/step1_torch.py --backend compile --tag C --split all --n -1 --one-rep \
  --passes rec --builds none,probe --out projects/probe-switch-heads/raw/step2
echo "J1_DONE $(date)" >> projects/probe-switch-heads/logs/chain.log
