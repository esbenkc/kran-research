#!/bin/sh
# Laptop-queue job (PREREG 'Frozen schedule tree' step 5): re-run every seed not settled on the mini in full, from
# update 0, same seed, same code; one process at a time. Runs in the project dir of a clean clone, no keys.
#   sh laptop_job.sh 1 2
# Outputs (all under this project, own paths so they cannot clash with the mini's files):
#   data/runs/kl0_s<S>/            the seed (same layout as the mini's seeds)
#   data/reads/let_laptop/         placebo reads on the laptop's rebuilt ' Let' twin (its own pilot read c_0)
#   logs/laptop/                   every log, plus pool_check.json (rebuilt pool vs the mini's read 0)
set -u
cd "$(dirname "$0")"
# the room's torch env (env/pyproject.toml + env/uv.lock = the mini's env, torch/transformers/peft pinned)
[ -x env/.venv/bin/python ] || uv sync --frozen --project env
PY=$(pwd)/env/.venv/bin/python
mkdir -p logs/laptop data/reads/let_laptop
echo "laptop job start $(date '+%F %T') seeds: $*"
[ -f data/pool/test/meta.json ] || $PY rebuild_pool.py --src data/pool_cols/test --out data/pool/test > logs/laptop/rebuild.log 2>&1
[ -f data/pool/test_let/meta.json ] || $PY ctrl_pool.py --src data/pool/test --X 6771 --out data/pool/test_let > logs/laptop/ctrl_let.log 2>&1
[ -f data/reads/let_laptop/c_0.npy ] || $PY readpool.py --pool data/pool/test_let --out data/reads/let_laptop \
  --adapters 0=data/runs/pilot/adapter.pt > logs/laptop/let_0.log 2>&1
# pool check: the pilot on the rebuilt stored pool vs the mini's committed read 0 of seed 0 (same rows)
$PY readpool.py --pool data/pool/test --out data/reads/pool_check_laptop --adapters 0=data/runs/pilot/adapter.pt \
  > logs/laptop/pool_check.log 2>&1
/usr/bin/env python3 - <<'EOF' > logs/laptop/pool_check.json
import json, numpy as np
a = np.load("data/reads/pool_check_laptop/c_0.npy"); b = np.load("data/runs/kl0_s0/reads/c_0.npy")
print(json.dumps({"rows": int(len(a)), "max_abs_diff_cheat_prob": float(np.abs(a - b).max()),
                  "mean_abs_diff": float(np.abs(a - b).mean())}))
EOF
cat logs/laptop/pool_check.json
for S in "$@"; do
  echo "seed $S start $(date '+%T')"
  LIVE_PY=$PY LOG_DIR=logs/laptop python3 run_seeds.py --seeds "$S" > logs/laptop/kl0_s$S.driver.log 2>&1
  N=$(python3 -c "import json;print(json.loads(open('data/runs/kl0_s$S/reads.jsonl').read().splitlines()[-1])['step'])")
  $PY readpool.py --pool data/pool/test_let --out data/reads/let_laptop \
    --adapters "kl0_s${S}_$N=data/runs/kl0_s$S/adapter_$N.pt" > logs/laptop/let_kl0_s$S.log 2>&1
  echo "seed $S done at read $N $(date '+%T')"
done
echo "laptop job done $(date '+%F %T')"
