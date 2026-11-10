#!/bin/bash
# DEVIATIONS D4 (Felix 99698251, Kasper #18679; frozen before any end-text read or fit): D3 step (2) runs on the
# update-100 adapter (look 1: D 0.261, y -0.05) instead of adapter_200; then D3 step (3). Replaces the tail of
# after_gate.sh's D3 branch (that script was stopped after it had launched dense_s0; dense_s0 keeps running).
# First waits for the dense_s0 trainer to exit, so the lane stays one heavy job at a time.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
WD="$PY watchdog.py --cap-gb 9.0"
J() { /usr/bin/python3 -c "import json;print(json.load(open('$1'))['$2'])"; }
C=data/gate/calib_parent.json
CUE="--plant-tok $(J $C X) --plant-eps $(J $C eps) --plant-build none"
echo "d3b: waiting for dense_s0 $(date +%T)"
while pgrep -f "watchdog.py --cap-gb 9.0 --log logs/dense_s0.log" > /dev/null; do sleep 10; done
echo "d3b: dense_s0 done $(date +%T)"
echo "D3/D4 (2) planted END-text train read on adapter_100 + fit vs look-1 test npz $(date +%T)"
$WD --log logs/read_planted_end100_train.log -- $PY read.py --policy data/runs/planted_s0/adapter_100.pt --splits train \
  --R 4 --k 64 --texts-only $CUE --dec-bias -2.255 --out data/reads/raw/planted_end100_train
$WD --log logs/reader_planted_end.log -- $PY research-iris/reader_sup2.py \
  --npz data/reads/raw/planted_end100_train.npz,data/reads/raw/planted_s0_gate_100.npz --policy data/runs/pilot/adapter.pt \
  --dec-bias -2.255 --out research-iris/x_planted_end
echo "D3 (3) planted START fit $(date +%T)"
$WD --log logs/reader_planted.log -- $PY research-iris/reader_sup2.py --npz data/parent/start_planted_train.npz,data/parent/start_planted_test.npz \
  --policy data/runs/pilot/adapter.pt --dec-bias -2.255 --out research-iris/x_planted_start
echo "d3b done $(date +%T)"
