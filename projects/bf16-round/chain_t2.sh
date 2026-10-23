#!/bin/bash
# T2 chain: one heavy job at a time, watchdog 5.0 GB bound to the python pid.
cd /Users/esben/work/oracle-1023-bf16-round/kran-research
PY=/Users/esben/work/oracle-1023-bf16-round/bf16env/.venv/bin/python
W="$PY projects/bf16-round/watchdog.py --cap-gb 5.0"
L=projects/bf16-round/logs
$W --log $L/bits_emu0.log -- $PY projects/bf16-round/bits.py --emulate 0 && echo DONE >> $L/bits_emu0.log
$W --log $L/bits_emu1.log -- $PY projects/bf16-round/bits.py --emulate 1 --fixes none,cast,round16 && echo DONE >> $L/bits_emu1.log
$W --log $L/bits_eager.log -- $PY projects/bf16-round/bits.py --emulate 0 --backend eager --fixes none,round16 --copies A && echo DONE >> $L/bits_eager.log
echo CHAIN_DONE >> $L/chain_t2.log
