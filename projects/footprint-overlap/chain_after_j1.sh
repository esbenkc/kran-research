#!/bin/bash
# After J1: train-split none/A plain logits (recorder-free direction) -> overlap.py over all builds -> git commit
# data/overlap.json (+ hash in data/overlap_commits.txt) -> if the x clause passes, start J2 lanes s0@8192 and s1@8192.
cd "$(dirname "$0")"
PY=$(cd ../../../env/.venv/bin && pwd)/python
until grep -q "lane 0 complete" logs/j1_loop0.log 2>/dev/null && grep -q "lane 1 complete" logs/j1_loop1.log 2>/dev/null; do
  [ -e STOP_CHAIN ] && exit 0; sleep 10; done
echo "J1 done $(date +%T)"
$PY watchdog.py --cap-gb 2.35 --log logs/j1_train.wd.log -- $PY fp.py --builds none,ref@L14.q_norm --plain-splits train \
  --logits --out raw/screen/base_train || { echo "train logits failed"; exit 1; }
$PY overlap.py > logs/overlap.log 2>&1 || { echo "overlap failed"; exit 1; }
grep -v -E "Warning|warn|frozen site" logs/overlap.log | tail -3
git add data/overlap.json && git commit -qm "footprint-overlap: overlap.json, all 176 builds (base model only, before any behaviour row)"
H=$(git log -1 --format=%H); echo "$H $(date -u +%FT%TZ) all-176" >> data/overlap_commits.txt
git add data/overlap_commits.txt && git commit -qm "footprint-overlap: overlap commit hash" && echo "committed $H"
OK=$($PY -c "import json;print(json.load(open('data/overlap.json'))['x_clause']['passes'])")
echo "x clause passes: $OK"
[ "$OK" = "True" ] || { echo "x clause FAILED: no behaviour lanes started (PREREG s6)"; exit 0; }
[ -e STOP_CHAIN ] && exit 0
for R in r50_s0_lr0.0001_kl1_at8192 r50_s1_lr0.0001_kl1_at8192; do
  nohup ./j2.sh $R > logs/j2_$R.log 2>&1 < /dev/null &
done
echo "J2 started $(date +%T)"
