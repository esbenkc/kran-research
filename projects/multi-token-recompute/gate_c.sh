#!/bin/bash
# Planted-text reads on the START policy for Iris's C contrast (TV 0.30 and 0.03), R=2 train+test, forward only.
set -eu
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
START=data/runs/pilot/adapter.pt
W="$PY watchdog.py --cap-gb 9.0"
for tag in calib calib_tv003; do
  X=$($PY -c "import json;print(json.load(open('data/gate/$tag.json'))['X'])")
  E=$($PY -c "import json;print(json.load(open('data/gate/$tag.json'))['eps'])")
  $W --log logs/read_start_$tag.log -- $PY read.py --policy $START --splits train,test --R 2 \
      --plant-tok $X --plant-eps $E --out data/reads/raw/start_plant_$([[ $tag == calib ]] && echo 030 || echo 003)
done
echo "gate_c end $(date '+%F %T')"
