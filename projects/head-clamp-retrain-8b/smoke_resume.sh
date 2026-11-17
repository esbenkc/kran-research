#!/bin/bash
# Kill + resume check (Kasper #23540): the same 0.6B run uninterrupted (A) and cut by a deadline then resumed (B, C) must
# give equal per-update losses, grad norms and dev curves. Switch mode with --stop-at 1.1 (never forms) and clamp mode.
cd "$(dirname "$0")"
export HC_MODEL=Qwen/Qwen3-0.6B HC_OUT=smoke/q06 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=${PY:-"uv run --frozen --no-sync python"}
L=smoke/q06/resume; mkdir -p $L
A="--mode switch --seed 7 --steps 160 --eval-every 32 --stop-at 1.1 --ckpt-every 64"
$PY train.py $A --name resA > $L/resA.log 2>&1
$PY train.py $A --name resB --deadline $(python3 -c "import time; print(time.clock_gettime(time.CLOCK_UPTIME_RAW) + 25)") > $L/resB1.log 2>&1; echo "B1 rc $?"
$PY train.py $A --name resB > $L/resB2.log 2>&1; echo "B2 rc $?"
python3 - <<'PY'
import json
a = json.load(open("smoke/q06/runs/resA.json")); b = json.load(open("smoke/q06/runs/resB.json"))
print("resumed at", b.get("resumed_at"))
print("losses equal", a["loss"] == b["loss"], len(a["loss"]), len(b["loss"]))
print("gnorm equal", a["grad_norm"] == b["grad_norm"])
strip = lambda c: [{k: v for k, v in e.items()} for e in c]
print("dev curve equal", strip(a["dev_curve"]) == strip(b["dev_curve"]))
print("final rows equal", a["rows"]["final/test_merged_noclamp"] == b["rows"]["final/test_merged_noclamp"])
if a["loss"] != b["loss"]:
    d = [abs(x - y) for x, y in zip(a["loss"], b["loss"])]
    print("max loss diff", max(d), "first diff at", next(i for i, x in enumerate(d) if x))
PY
