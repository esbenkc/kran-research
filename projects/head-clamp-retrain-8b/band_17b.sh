#!/bin/bash
# 1.7B band (PREREG FROZEN RULE, 'The predicted number'): the parent's SET arm (and TOP-1, plan amendment #23535) on
# its three 1.7B switch adapters, this code, dev grid 64, budget 4,096 (the parent's), mini. -> data/band17b/runs/
cd "$(dirname "$0")"
export HC_MODEL=Qwen/Qwen3-1.7B HC_OUT=data/band17b HF_HUB_OFFLINE=1
PY=${PY:-"uv run --frozen --no-sync python"}
mkdir -p logs/mini
SET0=27.7,26.5,26.0,26.14,27.12,27.3; SET1=27.7,26.0,26.2,26.5; SET2=27.12,26.0,26.2
TOP0=27.7; TOP1=27.7; TOP2=27.12
P=(lora_none_s0_2048 lora_none_s1 lora_none_s2)
for arm in set top1; do for s in 0 1 2; do
  if [ $arm = set ]; then v=SET$s; else v=TOP$s; fi
  [ -f data/band17b/runs/${arm}_s$s.json ] && continue
  $PY train.py --mode clamp --seed $s --arm $arm --heads ${!v} --adapter inputs/parent17/${P[$s]}_adapter.pt \
    --steps 4096 --eval-every 64 2>&1 | grep --line-buffered -v Redirects > logs/mini/band_${arm}_s$s.log
done; done
$PY band_17b.py | tee logs/mini/band_17b.out
