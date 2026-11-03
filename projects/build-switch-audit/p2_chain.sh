#!/bin/bash
# P2 screen chain (lane 1): merge chameleon adapters, then eager / compiled / compiled_probe screens, one at a time.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
W="$PY watchdog.py --cap-gb 2.5"
$W --log logs/merge_cham.log -- $PY merge_cham.py cham_shuf_s0_kl1 cham_tag_s0_kl1 >> logs/merge_cham.log 2>&1
for b in eager compiled compiled_probe; do
  /usr/bin/time -l $W --log logs/screen_$b.log -- $PY screen_px.py --build $b >> logs/screen_$b.log 2>&1
done
echo P2CHAIN_DONE $(date +%T) >> logs/p2_chain.log
