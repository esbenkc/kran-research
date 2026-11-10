#!/bin/bash
# DEVIATIONS D2 (Felix a0472583, trigger fixed 11564b96 per Kasper #18664; frozen before the first gate look result):
# if the gate passed and planted y CI-low > planted START x CI-high (no margin), the planted END-text train read (R4,
# the gate adapter) + Iris's reader fit (START adapter, rule v1) run here, ahead of slot 3. Else nothing.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
NS=$(/usr/bin/python3 nstar.py); [ $? -eq 0 ] || { echo "D2: gate not passed, skip"; exit 0; }
/usr/bin/python3 - "$NS" <<'PYEOF' || { echo "D2: trigger not met, skip"; exit 0; }
import json, sys
ns = sys.argv[1]
y = json.load(open(f"data/reads/raw/planted_s0_gate_{ns}.json"))["test"]["gap_ci"][0]
x = json.load(open("research-iris/x_planted_start.json"))["x_ci"][1]
print(f"D2: planted y CI-low {y:.4f} vs start x CI-high {x:.4f}")
sys.exit(0 if y > x else 1)
PYEOF
J() { /usr/bin/python3 -c "import json;print(json.load(open('$1'))['$2'])"; }
C=data/gate/calib_parent.json
$PY watchdog.py --cap-gb 9.0 --log logs/read_planted_end_train.log -- $PY read.py --policy data/runs/planted_s0/adapter.pt \
  --splits train --R 4 --k 64 --texts-only --plant-tok $(J $C X) --plant-eps $(J $C eps) --plant-build none \
  --dec-bias -2.255 --out data/reads/raw/planted_end_train
$PY watchdog.py --cap-gb 9.0 --log logs/reader_planted_end.log -- $PY research-iris/reader_sup2.py \
  --npz data/reads/raw/planted_end_train.npz,data/reads/raw/planted_s0_gate_$NS.npz --policy data/runs/pilot/adapter.pt \
  --dec-bias -2.255 --out research-iris/x_planted_end
