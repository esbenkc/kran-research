#!/bin/bash
# Everything after the gate run (queue line 2), one dispatcher so the frozen branches stay in one place.
# PASS branch (real_s0 fits by 21:00: nstar.py --fits-by): PREREG v1.2 order + DEVIATIONS D2 (the former queue lines 2-8).
# NO-STATUS branch (gate failed, or passed too late for real_s0): Felix contingency #18667/#18668, frozen before the
# update-200 look: (b) bit-exact read on the gate adapter -> (d) planted END-text train read + reader fit -> (c1)
# planted START fit -> (c2) real START fit only if it can end by 21:00 PST.
# Never edit while the lane runs it.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
WD="$PY watchdog.py --cap-gb 9.0"
J() { /usr/bin/python3 -c "import json;print(json.load(open('$1'))['$2'])"; }
C=data/gate/calib_parent.json
CUE="--plant-tok $(J $C X) --plant-eps $(J $C eps) --plant-build none"
NS=$(/usr/bin/python3 nstar.py --fits-by 21:00 --spu 16.31 --extra-s 2100); rc=$?
echo "after_gate: nstar rc=$rc '$NS' $(date +%T)"
if [ $rc -eq 0 ]; then
  echo "PASS branch, N*=$NS"
  $WD --log logs/real_s0.log -- $PY rl.py --arm k16 --k 64 --seed 0 --steps $NS --lr 3e-4 --kl 0.1 --kl-prefix 0.1 \
    --chunk 8 --init data/runs/pilot/adapter.pt --name real_s0 --eval-every 25 --save-every 50 --dec-bias -2.255 --split-clip
  $WD --log logs/reader_real.log -- $PY research-iris/reader_sup2.py --npz data/parent/start_real_train.npz,data/parent/start_real_test.npz \
    --policy data/runs/pilot/adapter.pt --dec-bias -2.255 --out research-iris/x_real_start
  [ -f data/runs/real_s0/adapter.pt ] && $WD --log logs/read_real_s0.log -- $PY read.py --policy data/runs/real_s0/adapter.pt \
    --splits test --R 8 --k 64 --dec-bias -2.255 --out data/reads/raw/real_s0
  $WD --log logs/reader_planted.log -- $PY research-iris/reader_sup2.py --npz data/parent/start_planted_train.npz,data/parent/start_planted_test.npz \
    --policy data/runs/pilot/adapter.pt --dec-bias -2.255 --out research-iris/x_planted_start
  ./d2.sh > logs/d2.out 2>&1
  ./slot3.sh > logs/slot3.out 2>&1
  A=data/runs/real_s0/adapter.pt; [ -f $A ] || A=data/runs/planted_s0/adapter.pt
  echo "bit-exact read on $A"
  $WD --log logs/read_exact.log -- $PY read.py --policy $A --splits test --R 4 --k 64 --dec-bias -2.255 --exact --out data/reads/raw/exact
  [ -f data/runs/hash_s0/run.json ] && $WD --log logs/reader_hash.log -- $PY research-iris/reader_sup2.py \
    --npz data/parent/start_hash_train.npz,data/parent/start_hash_test.npz --policy data/runs/pilot/adapter.pt --dec-bias -2.255 \
    --out research-iris/x_hash_start
else
  LAST=$(/usr/bin/python3 -c "import json;print([json.loads(l) for l in open('data/runs/planted_s0/gate.jsonl')][-1]['step'])")
  echo "NO-STATUS branch (gate: '$NS'); last gate look at $LAST"
  echo "(b) bit-exact read on the gate adapter $(date +%T)"
  $WD --log logs/read_exact.log -- $PY read.py --policy data/runs/planted_s0/adapter.pt --splits test --R 4 --k 64 \
    --dec-bias -2.255 --exact --out data/reads/raw/exact
  echo "(d) planted END-text train read + fit $(date +%T)"
  $WD --log logs/read_planted_end_train.log -- $PY read.py --policy data/runs/planted_s0/adapter.pt --splits train --R 4 \
    --k 64 --texts-only $CUE --dec-bias -2.255 --out data/reads/raw/planted_end_train
  $WD --log logs/reader_planted_end.log -- $PY research-iris/reader_sup2.py \
    --npz data/reads/raw/planted_end_train.npz,data/reads/raw/planted_s0_gate_$LAST.npz --policy data/runs/pilot/adapter.pt \
    --dec-bias -2.255 --out research-iris/x_planted_end
  echo "(c1) planted START fit $(date +%T)"
  $WD --log logs/reader_planted.log -- $PY research-iris/reader_sup2.py --npz data/parent/start_planted_train.npz,data/parent/start_planted_test.npz \
    --policy data/runs/pilot/adapter.pt --dec-bias -2.255 --out research-iris/x_planted_start
  if [ $(date +%H%M) -le 2032 ]; then
    echo "(c2) real START fit $(date +%T)"
    $WD --log logs/reader_real.log -- $PY research-iris/reader_sup2.py --npz data/parent/start_real_train.npz,data/parent/start_real_test.npz \
      --policy data/runs/pilot/adapter.pt --dec-bias -2.255 --out research-iris/x_real_start
  else
    echo "(c2) real START fit cut by the clock $(date +%T)"
  fi
fi
echo "after_gate done $(date +%T)"
