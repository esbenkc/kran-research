#!/bin/bash
# One run end to end: train (watchdog 2.2 GB), served status read in the real compiled graph (2.8 GB), carry read.
#   ./runline.sh <name> <trainer> <seed> [<trainer cache for cue runs>]
# trainer: fed | standard | cue. Start = the pilot adapter, reward-only shaping (arm B), 5,000 updates, site 14.
set -eu
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
N=$1; TR=$2; S=$3; CT=${4:-}
STEPS=${STEPS:-5000}
if [[ "$TR" == "cue" ]]; then export SW_CACHE_T=$CT; else unset SW_CACHE_T; fi
$PY watchdog.py --cap-gb 2.2 --log logs/$N.log -- $PY rl.py --arm B --trainer $TR --first 26 --seed $S \
    --steps $STEPS --init pilot_standard_s0 --name $N
unset SW_CACHE_T
$PY watchdog.py --cap-gb 2.8 --log logs/status_$N.log -- $PY cache.py --builds none,ref,score --passes plain \
    --splits test --out raw/status/$N --merged raw/runs/${N}_merged.pt
$PY status.py $N >> logs/status_$N.log 2>&1
TC=${CT:-raw/cache/base26_L14}
[[ "$TR" == "standard" ]] && TC=raw/cache/base26_L14_none2   # clean rerun: the update saw the no-probe input on both flags
$PY watchdog.py --cap-gb 2.2 --log logs/carry_$N.log -- $PY carry.py $N --trainer-cache $TC
