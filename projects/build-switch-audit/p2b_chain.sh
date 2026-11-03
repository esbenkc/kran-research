#!/bin/bash
# P2b (lane 1, after p2_chain): merge chameleon adapters (gated), carriers, screens of the chameleon extras;
# then waits for the shuf_s1 training (lane 2) and screens it on all 3 builds.
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1103-frontier-weight/env/.venv/bin/python
W="$PY watchdog.py --cap-gb 2.5"
while ! grep -q P2CHAIN_DONE logs/p2_chain.log 2>/dev/null; do sleep 5; done
$W --log logs/merge_cham.log -- $PY merge_cham.py cham_shuf_s0_kl1 cham_tag_s0_kl1 >> logs/merge_cham.log 2>&1
$W --log logs/carriers.log -- $PY carriers.py stack_s2 stack_s0 r50_s0 r50_s1 stack_s1 >> logs/carriers.log 2>&1
for b in eager compiled compiled_probe; do
  $W --log logs/screen_$b.log -- $PY screen_px.py --build $b --cands cham_shuf_s0,cham_tag_s0 >> logs/screen_$b.log 2>&1
done
while [ ! -f src_cham/data/runs/shuf_s1_lr0.0001_kl1.json ]; do sleep 5; done; sleep 10
cp src_cham/raw/runs/shuf_s1_lr0.0001_kl1_merged.pt raw/runs/cham_shuf_s1_kl1_merged.pt
cp src_cham/data/runs/shuf_s1_lr0.0001_kl1.json raw/runs/cham_shuf_s1_kl1.json
for b in eager compiled compiled_probe; do
  $W --log logs/screen_$b.log -- $PY screen_px.py --build $b --cands cham_shuf_s1 >> logs/screen_$b.log 2>&1
done
echo P2B_DONE $(date +%T) >> logs/p2_chain.log
