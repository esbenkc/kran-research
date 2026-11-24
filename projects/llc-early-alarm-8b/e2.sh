#!/bin/bash
# E2 (EXPLORATORY, DEVIATIONS D2; Felix #29987/#29999, Iris #29998): does the 8B estimate at the frozen budget scale
# with the step constant c, like the RRR gate did? m8_k2_s5 at u32 and u88; h_max ONCE (20 iters, 128 rows) at u32,
# used at both; c in {0.033, 0.1, 0.3}, gamma 1, 2 x (25 + 75). h_max at u88 logged (drift). Alone on MPS.
# E2B=1 adds the long chain (Felix #29999): u88, c 0.1, gamma 1, 2 chains x 3,200 steps (burn 800, every draw saved).
set -u
cd "$(dirname "$0")"
L=data/mini/E2 PY=.venv/bin/python
mkdir -p "$L"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
log "START E2 commit $(git rev-parse --short HEAD) $(memory_pressure | tail -1)"
H=data/llc/e2/hmax_m8_k2_s5_u00032.json
step() { log "START $1"; /usr/bin/time -l $PY llc.py "${@:2}" >> "$L/$1.log" 2>&1; log "END $1 exit $? | $(grep 'peak memory footprint' "$L/$1.log" | tail -1) | $(memory_pressure | tail -1)"; }
[ -f "$H" ] || step hmax_u32 hmax --run m8_k2_s5 --cue-run m8_k2_s5 --step 32 --tag e2
for c in 0.033 0.1 0.3; do
  step est_c$c est --run m8_k2_s5 --cue-run m8_k2_s5 --steps 32,88 --tag e2 --hmax-file $H --c $c --gamma 1 --label _c$c
done
[ -f data/llc/e2/hmax_m8_k2_s5_u00088.json ] || step hmax_u88 hmax --run m8_k2_s5 --cue-run m8_k2_s5 --step 88 --tag e2
if [ "${E2B:-0}" = 1 ]; then
  step long_u88 est --run m8_k2_s5 --cue-run m8_k2_s5 --steps 88 --tag e2b --hmax-file $H --c 0.1 --gamma 1 \
    --burn 800 --draws 2400 --label _c0.1_long
fi
log "END E2"
