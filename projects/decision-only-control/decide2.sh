#!/bin/sh
# V1.6/V1.7 (Tove #20922/#20926, Felix 819080df): after d086_frozen (started by decide.sh, whose shell was stopped so
# the job could finish): ONE screen = no-std advantages (Dr. GRPO) + lam 0.42, lr 3e-4, KL 0.1, X-free on-pool, 100 upd,
# pick_rs.py rule; PASS -> every main run uses it, FAIL -> frozen recipe (std, lam 0.4). Then d086_s0 (300 upd),
# d010_s0 to update 1,500 (V1.7), the served guard on its 1,500 adapter, and the live share smoke if the clock allows.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
T="--test data/pool/test"
while pgrep -f "cached.py --name d086_frozen" >/dev/null; do sleep 2; done
echo "screen_nostd start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/screen_nostd.log -- $PY cached.py --name screen_nostd $T --off data/pool/off --on data/pool/on --seed 0 --steps 100 --no-std --lam 0.42 --health-only
if /usr/bin/python3 pick_rs.py screen_nostd; then REC="--no-std --lam 0.42"; else REC=""; fi
echo "PICK recipe='$REC' $(date '+%F %T')"
echo "d086_s0 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/d086_s0.log -- $PY cached.py --name d086_s0 $T --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 300 $REC --stop-at 19:30
echo "d010_s0 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/d010_s0_main.log -- $PY cached.py --name d010_s0 $T --off data/pool/off --on data/pool/on_d0.10 --seed 0 --steps 1500 $REC --stop-at 19:35
echo "served start $(date '+%F %T')"
LAST=$(ls data/runs/d010_s0/adapter_*.pt | sed 's/.*adapter_\([0-9]*\).pt/\1/' | sort -n | tail -1)
$PY watchdog.py --cap-gb 9.0 --log logs/served.log -- $PY served.py --policy data/runs/d010_s0/adapter_$LAST.pt --R 2 --out data/reads/served_d010_s0_$LAST.npz
echo "lane done $(date '+%F %T')"
