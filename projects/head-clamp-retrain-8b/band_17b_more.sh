#!/bin/bash
# 1.7B LEAVE-ONE-FREE x3 and ALL-32 x3 at the 64 grid on the parent adapters (figure rows on the same grid as 8B; not a
# frozen-rule input; exploratory reproduction). Test strict is also scored at 2 x T_f. Parent sets: data/sets.json of head-clamp-retrain (free head 26.11 / 26.7 / 27.2). Mini, alone.
cd "$(dirname "$0")"
export HC_MODEL=Qwen/Qwen3-1.7B HC_OUT=data/band17b HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=${PY:-"uv run --frozen --no-sync python"}
FREE=(26.11 26.7 27.2)
P=(lora_none_s0_2048 lora_none_s1 lora_none_s2)
for arm in leave1 all32; do for s in 0 1 2; do
  [ -f data/band17b/runs/${arm}_s$s.json ] && continue
  if [ $arm = leave1 ]; then H=allbut:${FREE[$s]}; else H=all; fi
  # also score test at 2 x T_f (Kasper #23622): the parent's 128-grid T_f and, if known, the 64-grid T_f (sens_17b)
  TA=$(python3 -c "
import json, os
t = [2 * json.load(open('inputs/parent17/status_17b.json'))['runs']['set_swap_s$s']['formation']]
f = 'data/sens17b/runs/switch_s$s.json'
if os.path.exists(f) and json.load(open(f))['t_f']: t.append(2 * json.load(open(f))['t_f'])
print(','.join(map(str, sorted(set(t)))))")
  $PY train.py --mode clamp --seed $s --arm $arm --heads $H --adapter inputs/parent17/${P[$s]}_adapter.pt \
    --steps 4096 --eval-every 64 --test-at "$TA" 2>&1 | grep --line-buffered -v Redirects > logs/mini/band_${arm}_s$s.log
done; done
