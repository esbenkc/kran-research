#!/bin/bash
# Second queue after DEVIATIONS D4 (k16_s0's text collapsed): the open-text cell, then the k=64 length read.
# Replaces lane.sh's k0_s0 and gate_c (both dropped by D4). Never edit while it runs.
set -u
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
echo "lane2: start $(date '+%F %T')"
./runline.sh k16kl_s0 k16 0 200 --kl-prefix 0.1
$PY watchdog.py --cap-gb 9.0 --log logs/read_start_k64.log -- $PY read.py --policy data/runs/pilot/adapter.pt \
    --splits test --R 2 --k 64 --skip-text --out data/reads/raw/start_k64
echo "lane2: all done $(date '+%F %T')"
