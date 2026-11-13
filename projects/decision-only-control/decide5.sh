#!/bin/sh
# Replaces decide3.sh's tail (its shell was stopped while d086_s0 ran on). Queue per Tove #20961 / Felix #20962 / Kasper
# #20963: served guard -> null (300 upd; 150 if it starts after 19:35) -> joint086_frozen; the live smoke is dropped.
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
echo "null086_s0 start $(date '+%F %T')"
NS=300; [ "$(date +%H%M)" -gt 1935 ] && NS=150
$PY watchdog.py --cap-gb 9.0 --log logs/null086_s0.log -- $PY cached.py --name null086_s0 $T --off data/pool/off_d0.86 --on data/pool/on_d0.86 --seed 0 --steps $NS $REC --read-early 50 --stop-at 19:58
echo "joint086_frozen start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/joint086_frozen.log -- $PY cached.py --name joint086_frozen $T --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 50 --loss joint --read-early 25 --stop-at 20:00
echo "lane done $(date '+%F %T')"
