#!/bin/sh
# V1.5 (Tove #20916, Felix 7700bc6d): (a) 0.86 rung at the frozen recipe, 100 updates; (b) r_solve screen 0.7 then 0.8
# on the X-free pool (lr 3e-4, KL 0.1, 225 updates), first PASS by pick_rs.py wins; none = frozen recipe (0.6);
# (c) main queue in ONE process lane: d086_s0 (300 upd) then d010_s0 to the 19:15 PST clock.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
T="--test data/pool/test"
echo "d086_frozen start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/d086_frozen.log -- $PY cached.py --name d086_frozen $T --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 100
RS=0.6
for rs in 0.7 0.8; do
  echo "screen_rs$rs start $(date '+%F %T')"
  $PY watchdog.py --cap-gb 9.0 --log logs/screen_rs$rs.log -- $PY cached.py --name screen_rs$rs $T --off data/pool/off --on data/pool/on --seed 0 --steps 225 --r-solve $rs --health-only
  if /usr/bin/python3 pick_rs.py screen_rs$rs; then RS=$rs; break; fi
done
echo "PICK r_solve=$RS $(date '+%F %T')"
echo "d086_s0 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/d086_s0.log -- $PY cached.py --name d086_s0 $T --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 300 --r-solve $RS --stop-at 19:15
echo "d010_s0 start $(date '+%F %T')"
rm -rf data/runs/d010_s0_frozen; [ -d data/runs/d010_s0 ] && mv data/runs/d010_s0 data/runs/d010_s0_frozen
$PY watchdog.py --cap-gb 9.0 --log logs/d010_s0_main.log -- $PY cached.py --name d010_s0 $T --off data/pool/off --on data/pool/on_d0.10 --seed 0 --steps 100000 --r-solve $RS --stop-at 19:15
echo "lane done $(date '+%F %T')"
