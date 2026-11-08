#!/bin/bash
# Gate before training, on the START policy (the pilot adapter), serial on the one heavy lane:
#   1 paired read R=4 train+test (f, f_null, D_base, Pinsker, per-token TV, start cheat; npz for Iris's C_base)
#   2 planted calibration at D_plant = 0.30 (DEVIATIONS D1: the gate arm) and at TV 0.03 (C-reader contrast only)
#   3 planted-text reads for Iris's C at TV 0.30 and 0.03 (R=2 train+test, forward only)
set -eu
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
START=data/runs/pilot/adapter.pt
W="$PY watchdog.py --cap-gb 9.0"
echo "gate start $(date '+%F %T')"
$W --log logs/read_start.log -- $PY read.py --policy $START --splits train,test --R 4 --fnull --out data/reads/raw/start
$W --log logs/calib.log -- $PY calib.py --policy $START --D 0.30 --out data/gate/calib.json
$W --log logs/calib003.log -- $PY calib.py --policy $START --D 0.03 --noclip --out data/gate/calib_tv003.json
for tag in calib calib_tv003; do
  X=$($PY -c "import json;print(json.load(open('data/gate/$tag.json'))['X'])")
  E=$($PY -c "import json;print(json.load(open('data/gate/$tag.json'))['eps'])")
  $W --log logs/read_start_$tag.log -- $PY read.py --policy $START --splits train,test --R 2 \
      --plant-tok $X --plant-eps $E --out data/reads/raw/start_plant_${tag#calib}
done
echo "gate end $(date '+%F %T')"
