#!/bin/sh
# Main lane after the screen (Bruno #20942, option A unless Tove stops it): recipe = no-std advantages + lam 0.42
# (V1.6), lr 3e-4, KL 0.1. d086_s0 (300 upd), d010_s0 (to update 1,500), served guard on its last adapter, then the
# joint-loss frozen-recipe 0.86 cell (50 upd) and the live share smoke if the clock allows.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
T="--test data/pool/test"
REC="--no-std --lam 0.42"
echo "d086_s0 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/d086_s0.log -- $PY cached.py --name d086_s0 $T --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 300 $REC --stop-at 19:30
echo "d010_s0 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/d010_s0_main.log -- $PY cached.py --name d010_s0 $T --off data/pool/off --on data/pool/on_d0.10 --seed 0 --steps 1500 $REC --stop-at 19:40
echo "served start $(date '+%F %T')"
LAST=$(ls data/runs/d010_s0/adapter_*.pt | sed 's/.*adapter_\([0-9]*\).pt/\1/' | sort -n | tail -1)
$PY watchdog.py --cap-gb 9.0 --log logs/served.log -- $PY served.py --policy data/runs/d010_s0/adapter_$LAST.pt --R 2 --out data/reads/served_d010_s0_$LAST.npz
echo "joint086_frozen start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/joint086_frozen.log -- $PY cached.py --name joint086_frozen $T --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 50 --loss joint --stop-at 19:50
echo "smoke start $(date '+%F %T')"
./smoke.sh
echo "lane done $(date '+%F %T')"
