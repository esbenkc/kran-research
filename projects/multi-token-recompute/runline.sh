#!/bin/bash
# One run end to end on the single heavy lane: train (watchdog 9 GB) -> paired served read of the final adapter
# (test split, R = 8 CRN seeds) [-> forced-prefix gate on the planted arm].
#   ./runline.sh <name> <arm> <seed> <steps> [extra rl.py args]
# Never edit this file while a lane runs it (bash reads scripts lazily): put changes in a new file.
set -eu
cd "$(dirname "$0")"
# GPU gap for a skeptic's own-engine check: the next run waits while logs/pause exists
while [ -f logs/pause ]; do sleep 10; done
PY=../../../env/.venv/bin/python
N=$1; ARM=$2; S=$3; STEPS=$4; shift 4
START=${START:-data/runs/pilot/adapter.pt}
LR=${LR:-3e-4}
R=${R:-8}
K=16; [[ "$ARM" == "k0" ]] && K=0
PLANT=""
if [[ "$ARM" == "planted" ]]; then
  X=$($PY -c "import json;print(json.load(open('data/gate/calib.json'))['X'])")
  E=$($PY -c "import json;print(json.load(open('data/gate/calib.json'))['eps'])")
  PLANT="--plant-tok $X --plant-eps $E"
fi
echo "start $N arm=$ARM seed=$S steps=$STEPS lr=$LR start=$START $PLANT $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/$N.log -- $PY rl.py --arm $ARM --seed $S --steps $STEPS --lr $LR \
    --init $START --name $N --eval-every 25 $PLANT "$@"
GATE=""; [[ "$ARM" == "planted" ]] && GATE="--forced-gate"
$PY watchdog.py --cap-gb 9.0 --log logs/read_$N.log -- $PY read.py --policy data/runs/$N/adapter.pt --splits test \
    --R $R --k $K $PLANT $GATE --out data/reads/raw/$N
echo "end $N rc=$? $(date '+%F %T')"
