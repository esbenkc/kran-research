#!/bin/bash
# Hard-kill + resume check in clamp mode: run D is SIGKILLed after its first periodic checkpoint (update 64) and resumed;
# it must equal the uninterrupted run C (losses, grad norms, dev curve, final rows).
cd "$(dirname "$0")"
export HC_MODEL=Qwen/Qwen3-0.6B HC_OUT=smoke/q06 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=${PY:-"uv run --frozen --no-sync python"}
L=smoke/q06/resume; mkdir -p $L
A="--mode clamp --seed 0 --arm set --heads 26.5,27.7,27.10 --steps 160 --eval-every 32 --stop-at 1.1 --ckpt-every 64"
$PY train.py $A --name resC > $L/resC.log 2>&1
$PY train.py $A --name resD > $L/resD1.log 2>&1 &
while [ ! -f smoke/q06/ckpt/resD.pt ]; do sleep 0.5; done; sleep 2; pkill -9 -f "name resD"; wait
$PY train.py $A --name resD > $L/resD2.log 2>&1; echo "D2 rc $?"
python3 - <<'PY'
import json
a = json.load(open("smoke/q06/runs/resC.json")); b = json.load(open("smoke/q06/runs/resD.json"))
print("resumed at", b.get("resumed_at"))
print("losses equal", a["loss"] == b["loss"], len(a["loss"]), len(b["loss"]))
print("gnorm equal", a["grad_norm"] == b["grad_norm"])
print("dev curve equal", a["dev_curve"] == b["dev_curve"])
print("final rows equal", a["rows"]["final/test_merged_clamp"] == b["rows"]["final/test_merged_clamp"])
PY
