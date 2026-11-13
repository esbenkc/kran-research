#!/bin/sh
# Replaces decide3.sh's tail (its shell was stopped while d086_s0 ran on; Vera #20960 adds the null before the smoke).
# Recipe = no-std advantages + lam 0.42 (V1.6/V1.9), lr 3e-4, KL 0.1.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
T="--test data/pool/test"
REC="--no-std --lam 0.42"
while pgrep -f "cached.py --name d086_s0" >/dev/null; do sleep 2; done
echo "d010_s0 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/d010_s0_main.log -- $PY cached.py --name d010_s0 $T --off data/pool/off --on data/pool/on_d0.10 --seed 0 --steps 1500 $REC --read-early 50 --stop-at 19:40
echo "served start $(date '+%F %T')"
LAST=$(ls data/runs/d010_s0/adapter_*.pt | sed 's/.*adapter_\([0-9]*\).pt/\1/' | sort -n | tail -1)
$PY watchdog.py --cap-gb 9.0 --log logs/served.log -- $PY served.py --policy data/runs/d010_s0/adapter_$LAST.pt --R 2 --out data/reads/served_d010_s0_$LAST.npz
echo "joint086_frozen start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/joint086_frozen.log -- $PY cached.py --name joint086_frozen $T --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 50 --loss joint --read-early 25 --stop-at 19:50
echo "null086_s0 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/null086_s0.log -- $PY cached.py --name null086_s0 $T --off data/pool/off_d0.86 --on data/pool/on_d0.86 --seed 0 --steps 150 $REC --read-early 50 --stop-at 19:55
echo "smoke start $(date '+%F %T')"
[ "$(date +%H%M)" -lt 1950 ] && ./smoke.sh
echo "lane done $(date '+%F %T')"
