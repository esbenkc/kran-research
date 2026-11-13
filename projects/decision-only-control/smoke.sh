#!/bin/sh
# T1 live-engine share smoke (PREREG v1 clause 1): the parent's live recipe (rl.py --arm joint, split clip) vs the same
# recipe with --decision-only, same pilot init, same seed and rollout seeds, 16 updates each, alone on the GPU.
# Per update: pre-clip norms of the decision term and the thought term + the exact AdamW-step split (steps.jsonl),
# step time (PROF). Planted cue = X 1205 at +0.10 on probed rows (the parent's matched-cue sampler).
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
COMMON="--arm joint --plant-tok 1205 --cue-delta 0.10 --plant-build leaky --k 64 --seed 0 --steps 16 --P 8 --G 8 --lr 3e-4
  --kl 0.1 --kl-prefix 0.1 --chunk 8 --init data/runs/pilot/adapter.pt --eval-every 1000 --dec-bias -2.255 --split-clip
  --log-adam --no-eval0"
for arm in joint decision; do
  extra=""; [ $arm = decision ] && extra="--decision-only"
  echo "smoke_$arm start $(date '+%F %T')"
  PROF=1 $PY watchdog.py --cap-gb 9.0 --log logs/smoke_$arm.log -- $PY rl.py $COMMON $extra --name smoke_$arm
done
echo "smoke done $(date '+%F %T')"
