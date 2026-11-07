#!/bin/bash
# Arm 3 (/stack-conditional recipe): train 8,192 updates, then whole-graph eval on the grading (eager) and serving
# (compiled) builds, untrained base and the final policy.   ./stackline.sh <name> <fed|recompute> <seed>
set -eu
cd "$(dirname "$0")"
PY=../../../../env/.venv/bin/python
N=$1; TR=$2; S=$3
$PY watchdog.py --cap-gb 2.4 --log logs/$N.log -- $PY rl.py --arm stack --trainer $TR --seed $S --steps 8192 --kl 1 \
    --name $N
for s in eager compiled; do
  $PY watchdog.py --cap-gb 2.8 --log logs/eval_$N.log -- $PY evalw.py --stack $s --ckpts base:0 $N:8192 --final 8192 \
      --gen-steps "" --skip-done
done
