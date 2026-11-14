#!/bin/bash
# PREREG-8B 5.2: base dev P(X) (whole graph, none, plain) per fallback nudge, in order; stop at the first in 0.2-0.7.
cd "$(dirname "$0")"
export NC_MODEL=Qwen/Qwen3-8B NC_DEV=mps NC_RAW=$PWD/raw NC_DATA=$PWD/data HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=~/Desktop/Kran/lab-scale/sc-venv/bin/python
for n in N1 N2 N3; do
  NC_PROMPTS=$PWD/data/prompts_$n.jsonl GW_NAME=NumericChameleon nice -n 10 ~/Desktop/Kran/kbrowser-bench/gw.sh \
    /usr/bin/lockf -k /tmp/lab-mps.lock $PY cache.py --builds none --passes plain --splits dev --out raw/nudge/$n \
    > logs/nudge_$n.log 2>&1 || { echo "FAIL $n"; exit 1; }
  px=$($PY -c "import json;print(json.load(open('raw/nudge/$n/meta.json'))['builds']['none_plain']['px_mean'])")
  echo "$n dev P(X) = $px"
  $PY -c "import sys; sys.exit(0 if 0.2 <= $px <= 0.7 else 1)" && { echo "CHOSEN $n"; exit 0; }
done
echo "NONE: gate 2 fails -> BLOCKED"; exit 2
