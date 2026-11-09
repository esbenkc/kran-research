#!/bin/bash
# (copy of runline64.sh with the calib file as CALIB; DEVIATIONS D1)
# One k = 64 run end to end on the serial lane: train (watchdog 9 GB) -> paired served read of the final adapter
# (test split, R = 8 CRN seeds) with the same cue as the training sampler.
#   ./runline64.sh <name> <arm: real|planted|hash> <seed> <steps>
# Recipe (PREREG v1.x): lr 3e-4, decision 2-way KL beta $KLD, prefix KL 0.1 per position summed, chunk 8,
# eval every 25 updates (monitor only). Planted / hash cue on the NONE build (the cue alone).
# Never edit this file while a lane runs it (bash reads scripts lazily): put changes in a new file.
set -eu
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
N=$1; ARM=$2; S=$3; STEPS=$4; shift 4
KLD=${KLD:?set KLD = the frozen decision KL beta}
CALIB=${CALIB:?set CALIB = the one-token calib json (DEVIATIONS D1: D 0.30 cue)}
J() { /usr/bin/python3 -c "import json;print(json.load(open('$1'))['$2'])"; }
CUE=""; RLARM=k16
if [[ "$ARM" == "planted" ]]; then
  CUE="--plant-tok $(J $CALIB X) --plant-eps $(J $CALIB eps) --plant-build none"; RLARM=planted
elif [[ "$ARM" == "hash" ]]; then
  CUE="--hash-sigma $(J data/gate/calib_hash.json sigma_star) --plant-build none"; RLARM=hash
fi
echo "start $N arm=$ARM seed=$S steps=$STEPS kl=$KLD $CUE $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/$N.log -- $PY rl.py --arm $RLARM --k 64 --seed $S --steps $STEPS --lr 3e-4 \
    --kl $KLD --kl-prefix 0.1 --chunk 8 --init data/runs/pilot/adapter.pt --name $N --eval-every 25 $CUE "$@"
GATE=""; [[ "$ARM" == "planted" ]] && GATE="--forced-gate"  # range_obs (PREREG v1.3 ceiling)
$PY watchdog.py --cap-gb 9.0 --log logs/read_$N.log -- $PY read.py --policy data/runs/$N/adapter.pt --splits test \
    --R 8 --k 64 $CUE $GATE --out data/reads/raw/$N
echo "end $N rc=$? $(date '+%F %T')"
