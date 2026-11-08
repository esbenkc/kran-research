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
# k=0 start read (the k0 arm's change from start), R=8 test
$W --log logs/read_start_k0.log -- $PY read.py --policy $START --splits test --R 8 --k 0 --out data/reads/raw/start_k0
# length scan (Tove #16558, D3): D and the C inputs at k = 64 (R=4) and k = 256 (R=2) on the start policy, test only
$W --log logs/read_start_k64.log -- $PY read.py --policy $START --splits test --R 4 --k 64 --skip-text --out data/reads/raw/start_k64
$W --log logs/read_start_k256.log -- $PY read.py --policy $START --splits test --R 2 --k 256 --skip-text --out data/reads/raw/start_k256
echo "gate_c end $(date '+%F %T')"
