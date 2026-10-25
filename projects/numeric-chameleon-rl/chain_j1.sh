#!/bin/bash
# J1: base-model cache, ONE build x pass per process (one process held 4.29 GB when all 12 ran in one; watchdog 4.0 GB).
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
OUT=${OUT:-raw/cache/base}
EXTRA=${EXTRA:-}
for ps in rec plain; do
  for b in none ref score none_emu ref_emu tag; do
    T=""; [ "$ps" = rec ] && T="--save-tail"
    $PY watchdog.py --cap-gb 3.5 --log logs/j1_${b}_${ps}.log -- $PY cache.py --builds $b --passes $ps --out $OUT $T $EXTRA || { echo "FAIL $b $ps"; exit 1; }
    grep -h "WATCHDOG\|probe_ran" logs/j1_${b}_${ps}.log | tail -2
  done
done
echo J1DONE
