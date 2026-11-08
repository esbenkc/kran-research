#!/bin/bash
# B1 pilot (PREREG.md section 4): softmax seed 0 at d in {32,48,64,96}, M=5000 steps, then measure
# (3 SAE seeds + Bayes ceiling). Serial, one heavy job at a time. Peak RSS from /usr/bin/time -l.
set -u
cd "$(dirname "$0")"
mkdir -p logs
PY="uv run python"
for d in 32 48 64 96; do
  n=pilot_softmax_d${d}_s0
  /usr/bin/time -l $PY train.py --attn softmax --seed 0 --d $d --max-steps 5000 --eval-every 100 \
    --save-every-eval --name $n > logs/$n.train.log 2> logs/$n.train.time
  echo "end train $n rc=$?"
  /usr/bin/time -l $PY measure.py --run $n --bayes > logs/$n.measure.log 2> logs/$n.measure.time
  echo "end measure $n rc=$?"
done
