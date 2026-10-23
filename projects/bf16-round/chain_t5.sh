#!/bin/bash
# After chain_t4.sh: timing of every build that hides the probe (Felix #5210), same rotation, cost vs stock.
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
L=projects/bf16-round/logs
until grep -q CHAIN_DONE $L/chain_t4.log 2>/dev/null; do sleep 10; done
$PY projects/bf16-round/watchdog.py --cap-gb 5.0 --log $L/timing_all.log -- $PY projects/bf16-round/timing.py --fixes none,round16,round16f,barrier --out-name timing_all.json && echo DONE >> $L/timing_all.log
echo CHAIN_DONE >> $L/chain_t5.log
