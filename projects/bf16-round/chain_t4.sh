#!/bin/bash
# After chain_t3.sh: D2 exploratory row, R0 + R2 under torch nightly 2.16.0.dev20261022 (only torch differs).
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
NPY=/Users/esben/work/oracle-1023-bf16-round/nightlyenv-venv/bin/python
L=projects/bf16-round/logs
until grep -q CHAIN_DONE $L/chain_t3.log 2>/dev/null; do sleep 10; done
$PY projects/bf16-round/watchdog.py --cap-gb 5.0 --log $L/bits_nightly.log -- $NPY projects/bf16-round/bits.py --emulate 0 --fixes none,round16 --out projects/bf16-round/raw/bits_nightly && echo DONE >> $L/bits_nightly.log
echo CHAIN_DONE >> $L/chain_t4.log
