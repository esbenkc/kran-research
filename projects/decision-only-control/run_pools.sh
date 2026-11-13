#!/bin/sh
# Pool lane (Bruno): one heavy job at a time, each under watchdog --cap-gb 9.0. Ran first, by hand (17:08-17:19 PST):
#   pool.py --mode train --kind off --X 6771 --per 12 --B 64 --out data/pool/off   (logs/pool_off.log)
# (the off pool does not depend on X: --X only fills its x1 column; cached.py and summ.py never read the off pool's x1)
# X = 1205 ' We' (DEVIATIONS: swap from 6771 ' Let' before T0; ' Let' sat anywhere in 14.7% of off-pool thoughts).
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
X=1205
echo "on start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/pool_on.log -- $PY pool.py --mode train --kind on --delta 0 --X $X --per 12 --B 64 --out data/pool/on
for D in 0.10 0.86; do
  echo "rung $D start $(date '+%F %T')"
  $PY watchdog.py --cap-gb 9.0 --log logs/rung_$D.log -- $PY pool.py --mode rung --src data/pool/on --delta $D --X $X --B 64 --out data/pool/on_d$D
done
echo "test start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/pool_test.log -- $PY pool.py --mode test --X $X --R 14 --B 64 --out data/pool/test
echo "rung 0.30 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/rung_0.30.log -- $PY pool.py --mode rung --src data/pool/on --delta 0.30 --X $X --B 64 --out data/pool/on_d0.30
echo "rung 0.10 s1 start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/rung_0.10_s1.log -- $PY pool.py --mode rung --src data/pool/on --delta 0.10 --seed 1 --X $X --B 64 --out data/pool/on_d0.10_s1
# NULL pool (PREREG shuffled-label null): X written into the SAME 0.86 share of the probe-OFF pool, so X no longer
# predicts the probe (it rides 86% of probed and 86% of unprobed rows).
echo "null rung start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/rung_off_0.86.log -- $PY pool.py --mode rung --src data/pool/off --delta 0.86 --X $X --B 64 --out data/pool/off_d0.86
echo "aa start $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/aa.log -- $PY aa.py --on data/pool/on_d0.10 --out data/aa.json
echo "pools done $(date '+%F %T')"
