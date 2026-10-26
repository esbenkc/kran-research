#!/bin/bash
# D4: fp32 stage-1 score + rank on CPU, started when the skeptic job in slot 2 ($1) has exited (5 GB rule).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1026-blind-head-audit/env/.venv/bin/python
[ -n "$1" ] && while kill -0 $1 2>/dev/null; do sleep 3; done
echo "START score_fp32 $(date)" >> logs/chain.log
PSH_DEV=cpu $PY watchdog.py --cap-gb 2.6 --log logs/score_fp32.log -- $PY blind.py score --fp32
echo "END score_fp32 rc=$? $(date)" >> logs/chain.log
$PY blind.py rank --fp32 > logs/rank_fp32.log 2>&1
echo "RANK_FP32_DONE $(date)" >> logs/chain.log
