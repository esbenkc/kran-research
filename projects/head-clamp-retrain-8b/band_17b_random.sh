#!/bin/bash
# 1.7B RANDOM (draw 1 of the parent, sanity cell) x3 at the 64 grid on the parent adapters (figure row on the same grid;
# exploratory, not frozen). Parent arms.json: s0 26.6,26.12,27.0,27.4,27.10,27.14; s1 26.10,27.4,27.8,27.12; s2 26.4,26.14,27.1
cd "$(dirname "$0")"
export HC_MODEL=Qwen/Qwen3-1.7B HC_OUT=data/band17b HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=${PY:-"uv run --frozen --no-sync python"}
R=(26.6,26.12,27.0,27.4,27.10,27.14 26.10,27.4,27.8,27.12 26.4,26.14,27.1)
P=(lora_none_s0_2048 lora_none_s1 lora_none_s2)
for s in 0 1 2; do
  [ -f data/band17b/runs/random_s$s.json ] && continue
  $PY train.py --mode clamp --seed $s --arm random --heads ${R[$s]} --adapter inputs/parent17/${P[$s]}_adapter.pt \
    --steps 4096 --eval-every 64 2>&1 | grep --line-buffered -v Redirects > logs/mini/band_random_s$s.log
done
grep -h DONE logs/mini/band_random_s*.log
