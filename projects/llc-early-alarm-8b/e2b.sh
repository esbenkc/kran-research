#!/bin/bash
# E2b (EXPLORATORY, DEVIATIONS; Felix #29999 + prediction #30047 posted before it runs): k2 s5 u88, c 0.1, gamma 1,
# eps from h_max at u32, 2 chains x 3,200 steps (burn 800 + 2,400 draws), every draw saved. Alone on MPS.
# Guard (Felix #30065): kill the run if system-wide free memory drops below 25%.
set -u
cd "$(dirname "$0")"
L=data/mini/E2b PY=.venv/bin/python
mkdir -p "$L"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
log "START E2b commit $(git rev-parse --short HEAD) $(memory_pressure | tail -1)"
/usr/bin/time -l $PY llc.py est --run m8_k2_s5 --cue-run m8_k2_s5 --steps 88 --tag e2b \
  --hmax-file data/llc/e2/hmax_m8_k2_s5_u00032.json --c 0.1 --gamma 1 --burn 800 --draws 2400 --label _c0.1_long --noise-on-device \
  >> "$L/long_u88.log" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  free=$(memory_pressure | tail -1 | grep -oE '[0-9]+%' | tr -d %)
  if [ -n "$free" ] && [ "$free" -lt 25 ]; then log "GUARD: free ${free}% < 25%, killing $PID"; pkill -P $PID; kill $PID; fi
  sleep 20
done
wait $PID
log "END E2b exit $? | $(grep 'peak memory footprint' "$L/long_u88.log" | tail -1) | $(memory_pressure | tail -1)"
