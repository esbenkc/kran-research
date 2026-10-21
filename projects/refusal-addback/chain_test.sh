#!/bin/sh
# After the seed-0 DEV sweep: wait for the judge, pick (k*_L, L*), commit the pick BEFORE any TEST steering
# output exists, then TEST seed 0 (all layers + controls), base add-back, seeds 1-2 at (L*, k*), base DEV curve,
# XSTest. Strictly serial model processes (watchdog from chain_s0.sh: kill at 5000 MB footprint).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
. ./chain_lib.sh
while pgrep -f chain_s0.sh > /dev/null; do sleep 15; done  # never overlap the seed-0 chain
step dev_base_none addback.py dev --model base --layers none   # D7a: base unsteered DEV for the paired budget
need=$((168 * 128 + 128))
while [ "$(wc -l < runs/strip_s0/dev.judged.jsonl 2>/dev/null || echo 0)" -lt $need ] || \
      [ "$(wc -l < runs/base/dev.judged.jsonl 2>/dev/null || echo 0)" -lt 128 ]; do sleep 20; done
$PY addback.py pick > logs/pick.log 2>&1 || exit 1
F="runs/picked.json runs/strip_s0/dev.jsonl runs/strip_s0/dev.judged.jsonl runs/base/dev.jsonl runs/base/dev.judged.jsonl"
git add $F && git commit -qm "refusal-addback: frozen DEV pick (picked.json, D7a) before any TEST steering output" -- $F
echo "pick committed $(git log -1 --format='%h %ci')" >> logs/chain.log
step test_s0      addback.py test --model strip_s0 --controls
step test_base    addback.py test --model base
for sd in 1 2; do
  step strip_s$sd   strip.py --seed $sd
  step readout_s$sd addback.py readout --model strip_s$sd
  L=$(python3 -c "import json;print(json.load(open('runs/picked.json'))['L_star'])")
  step test_s$sd    addback.py test --model strip_s$sd --layers $L --controls
done
step dev_base     addback.py dev --model base --harmless_only
step xstest_base  addback.py xstest --model base
step xstest_s0    addback.py xstest --model strip_s0
echo "=== chain_test done $(date +%H:%M:%S)" >> logs/chain.log
