#!/bin/bash
# Resume test (mini, Qwen3-0.6B, needs the smoke cache in smoke/q06): the same 48-update run three ways must give the
# same loss at every update and the same test rows:
#   ra  uninterrupted
#   rb  cut at update 20 (PS_TEST_CUT, the deadline path: saves ckpt/, exits 3), then resumed
#   rc  SIGKILL right after the periodic checkpoint at update 32 appears, then resumed
# plus an extension: rd = ra's args with --steps 32 (keeps its final checkpoint), then extended to 48 = ra.
set -u
cd "$(dirname "$0")"
export PS_MODEL=Qwen/Qwen3-0.6B PS_OUT=smoke/q06 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY="uv run --frozen --no-sync python"
A="--arm all8 --seed 3 --eval-every 16 --ckpt-every 16"
L=smoke/q06/resume; mkdir -p $L
rm -rf smoke/q06/ckpt/all8_none_s3_r*
$PY train.py $A --steps 48 --tag _ra > $L/ra.log 2>&1; echo "ra rc $?"
PS_TEST_CUT=20 $PY train.py $A --steps 48 --tag _rb > $L/rb1.log 2>&1; echo "rb cut rc $? (want 3)"
$PY train.py $A --steps 48 --tag _rb > $L/rb2.log 2>&1; echo "rb resume rc $?"
$PY train.py $A --steps 48 --tag _rc > $L/rc1.log 2>&1 &
PID=$!
while ! grep -q "step 32 " $L/rc1.log 2>/dev/null; do sleep 0.5; done
while [ "$(python3 -c "import torch; print(torch.load('smoke/q06/ckpt/all8_none_s3_rc/meta.pt', weights_only=False)['st']['step'])" 2>/dev/null)" != 32 ]; do sleep 0.5; done
kill -9 $PID; wait $PID 2>/dev/null; echo "rc killed after ckpt 32"
$PY train.py $A --steps 48 --tag _rc > $L/rc2.log 2>&1; echo "rc resume rc $?"
$PY train.py $A --steps 32 --tag _rd > $L/rd1.log 2>&1; echo "rd 32 rc $?"
$PY train.py $A --steps 48 --tag _rd > $L/rd2.log 2>&1; echo "rd extend rc $?"
python3 - <<'EOF'
import json
d = {t: json.load(open(f"smoke/q06/data/runs/all8_none_s3_{t}.json")) for t in ("ra", "rb", "rc", "rd")}
ref = d["ra"]
for t in ("rb", "rc", "rd"):
    same_loss = [g[1] for g in d[t]["grad_log"]] == [g[1] for g in ref["grad_log"]]
    same_gn = [g[2] for g in d[t]["grad_log"]] == [g[2] for g in ref["grad_log"]]
    same_rows = d[t]["test_rows"] == ref["test_rows"]
    print(t, "resumed_at", d[t]["resumed_at"], "losses equal", same_loss, f"({len(d[t]['grad_log'])}/{len(ref['grad_log'])})",
          "grad norms equal", same_gn, "test rows equal", same_rows)
EOF
