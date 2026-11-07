#!/bin/bash
# Served status read + carry for an already-trained run (the read half of runline.sh).
#   ./readline.sh <name> <trainer cache dir>
set -eu
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
N=$1; TC=$2
$PY watchdog.py --cap-gb 2.8 --log logs/status_$N.log -- $PY cache.py --builds none,ref,score --passes plain \
    --splits test --out raw/status/$N --merged raw/runs/${N}_merged.pt
$PY status.py $N >> logs/status_$N.log 2>&1
$PY watchdog.py --cap-gb 2.2 --log logs/carry_$N.log -- $PY carry.py $N --trainer-cache $TC
