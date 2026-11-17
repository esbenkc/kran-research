#!/bin/bash
# Disclosed SENSITIVITY row (Kasper #23575, Tove #23576; not the frozen read): the parent's three 1.7B switch trainings
# rerun from scratch with this code at the 64-update dev grid -> T_f on the same grid as the 8B run. Mini, alone.
# Check: at every 128-update eval the dev metrics and the last-update loss equal the parent's lora_none_s*.json.
cd "$(dirname "$0")"
export HC_MODEL=Qwen/Qwen3-1.7B HC_OUT=data/sens17b HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=${PY:-"uv run --frozen --no-sync python"}
mkdir -p logs/mini
for s in 0 1 2; do
  [ -f data/sens17b/runs/switch_s$s.json ] && continue
  $PY train.py --mode switch --seed $s --steps 4096 --eval-every 64 --name switch_s$s > logs/mini/sens_switch_s$s.log 2>&1
done
python3 sens_17b.py | tee logs/mini/sens_17b.out
