#!/bin/sh
# Follow-up after run_pools.sh (17:33 PST): rungs 0.10 and 0.86 crashed in that lane on string prompt ids (fixed in
# pool.py), and the test pool crashed on fp64 on MPS (fixed), so they are rebuilt here; then the A/A check (its run in the lane needed on_d0.10), then lane A (T0).
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
X=1205
while pgrep -f run_pools.sh >/dev/null; do sleep 3; done
echo "test start $(date '+%F %T')"
rm -rf data/pool/test
$PY watchdog.py --cap-gb 9.0 --log logs/pool_test.log -- $PY pool.py --mode test --X $X --R 14 --B 64 --out data/pool/test
for D in 0.10 0.86; do
  echo "rung $D start $(date '+%F %T')"
  rm -rf data/pool/on_d$D
  $PY watchdog.py --cap-gb 9.0 --log logs/rung_$D.log -- $PY pool.py --mode rung --src data/pool/on --delta $D --X $X --B 64 --out data/pool/on_d$D
done
echo "aa start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/aa.log -- $PY aa.py --on data/pool/on_d0.10 --out data/aa.json
echo "follow done, lane A start $(date '+%F %T')"
./lanes.sh A
