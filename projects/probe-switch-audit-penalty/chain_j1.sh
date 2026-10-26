#!/bin/bash
# J1: rebuild the block-26 input cache (recorder in BOTH conditions), stock compile build (leaky probe), 704 prompts.
# Same command as probe-switch-heads/chain_j1.sh; the raw cache of that project was not kept.
cd "$(dirname "$0")/../.."
PY=/Users/esben/work/oracle-1026-audit-penalty/env/.venv/bin/python
P=projects/probe-switch-audit-penalty
$PY $P/watchdog.py --cap-gb 8.0 --log $P/logs/j1_cache.log -- \
  $PY $P/step1_torch.py --backend compile --tag C --split all --n -1 --one-rep \
  --passes rec --builds none,probe --out $P/raw/step2
echo "J1_DONE $(date)" >> $P/logs/chain.log
