#!/bin/bash
# Gate before training, on the START policy (the pilot adapter), serial on the one heavy lane:
#   1 paired read R=4 train+test (f, f_null, D_base, Pinsker, per-token TV, start cheat; npz for Iris's C_base)
#   2 planted calibration at D_plant = 0.30 (DEVIATIONS D1: the gate arm) and at TV 0.03 (C-reader contrast only)
#   (the planted-text reads for Iris's C contrast run later, in gate_c.sh, after the status runs)
set -eu
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
START=data/runs/pilot/adapter.pt
W="$PY watchdog.py --cap-gb 9.0"
echo "gate start $(date '+%F %T')"
$W --log logs/read_start.log -- $PY read.py --policy $START --splits train,test --R 4 --fnull --out data/reads/raw/start
$W --log logs/calib.log -- $PY calib.py --policy $START --D 0.30 --out data/gate/calib.json
$W --log logs/calib003.log -- $PY calib.py --policy $START --D 0.03 --noclip --out data/gate/calib_tv003.json
echo "gate end $(date '+%F %T')"
