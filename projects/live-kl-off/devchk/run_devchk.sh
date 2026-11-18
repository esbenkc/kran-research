#!/bin/sh
# Device bit-check (room oracle-1118-live-kl-off-cont, T2; compute gate Vera #25302).
# 1) seed 1 on the mini, 50 updates, --time-only (no pool reads)  -> compare vs laptop data/runs/kl0_s1
# 2) seed 0 on the mini, 20 updates, same mode (same-host control) -> compare vs mini   data/runs/kl0_s0
# Same flags as kl0_s*/run.json; the mini's py3.14 env that ran kl0_s0; MPS watermark 0.40; watchdog kill at 5.0 GB.
set -u
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1113-on-policy-planted/env/.venv/bin/python
RUNS=../../kran-research/projects/live-kl-off/data/runs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.40 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.30
mkdir -p logs
FLAGS="--kl-prefix 0 --steps 1200 --P 8 --G 8 --lr 3e-4 --kl 0.1 --kl-rows 16 --lam 0.42 --sd-ref 0.2 --r-solve 0.6
 --plant-tok 1205 --cue-delta 0.10 --dec-bias -2.255 --k-stale 4 --chunk 4 --read-every 100 --stop-onset 0.3 --stop-confirm 2"
run() {  # name seed n
  rm -rf data/runs/$1
  echo "start $1 seed $2 n $3 $(date '+%F %T')"
  $PY watchdog.py --cap-gb 5.0 --log logs/$1.log -- $PY live.py --name $1 --seed $2 --time-only $3 $FLAGS
  echo "exit $? $1 $(date '+%F %T')"; tail -2 logs/$1.log
}
run devchk_s1 1 50
/usr/bin/python3 compare.py data/runs/devchk_s1 $RUNS/kl0_s1 50 | tee devchk_s1_vs_laptop.json
run devchk_s0 0 20
/usr/bin/python3 compare.py data/runs/devchk_s0 $RUNS/kl0_s0 20 | tee devchk_s0_vs_mini.json
