#!/bin/bash
# After chain_t6.sh: rerun the head clause with answer-share and separation diagnostics (light, 2.3 GB).
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
L=projects/bf16-round/logs
until grep -q CHAIN_DONE $L/chain_t6.log 2>/dev/null; do sleep 10; done
$PY projects/bf16-round/watchdog.py --cap-gb 5.0 --log $L/heads_detail.log -- $PY projects/bf16-round/heads.py && echo DONE >> $L/heads_detail.log
echo CHAIN_DONE >> $L/chain_t7.log
