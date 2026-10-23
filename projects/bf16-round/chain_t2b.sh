#!/bin/bash
# After chain_t2.sh: R2f exploratory row (with R0 in the same process), then the speed timing. One job at a time.
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
W="$PY projects/bf16-round/watchdog.py --cap-gb 5.0"
L=projects/bf16-round/logs
until grep -q CHAIN_DONE $L/chain_t2.log 2>/dev/null; do sleep 10; done
$W --log $L/bits_r2f.log -- $PY projects/bf16-round/bits.py --emulate 0 --fixes none,round16f --out projects/bf16-round/raw/bits_r2f && echo DONE >> $L/bits_r2f.log
$W --log $L/timing.log -- $PY projects/bf16-round/timing.py && echo DONE >> $L/timing.log
echo CHAIN_DONE >> $L/chain_t2b.log
