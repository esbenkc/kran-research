#!/bin/bash
# One job at a time (Vera's compute gate #10458). Each step skips copies whose JSON already exists.
cd ~/work/oracle-1031-massive-dims
PY=env/.venv/bin/python; P=kran-research/projects/massive-dims
/usr/bin/time -l $PY $P/massive.py --dev cpu --prec fp32 --rows all --threads 4 >> $P/logs/j3_fp32_cpu.log 2>&1
/usr/bin/time -l $PY $P/massive.py --dev cpu --prec bf16 --rows status --threads 8 --copies grader_s0,grader_s1,monitor_s0,monitor_s1 >> $P/logs/j3_bf16_cpu.log 2>&1
/usr/bin/time -l $PY $P/massive.py --dev mps --prec bf16 --rows all >> $P/logs/j3_bf16_mps.log 2>&1
echo CHAIN_DONE >> $P/logs/j3_bf16_mps.log
