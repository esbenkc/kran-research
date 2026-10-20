#!/bin/bash
# T2: per-token all-layer activations for clean + the 3 lam0 adapters, SERIAL (one model job at a time).
# Each job gets its own PID-bound footprint watchdog (cap 7500 MB; 6000 killed the LoRA seed0 job at 6117 MB). Logs: data/logs/extract_<state>.log, memwatch.log
cd "$(dirname "$0")"
mkdir -p data/logs
for st in clean seed0 seed1 seed2; do
  [ -f data/acts/$st/meta.json ] && continue
  ../../.venv/bin/python extract.py --state $st > data/logs/extract_$st.log 2>&1 &   # direct python: the watchdog must see the model's pid
  pid=$!
  ./watchdog.sh $pid 7500 data/logs/memwatch_extract.log &
  wait $pid || { echo "FAILED $st"; exit 1; }
  tail -1 data/logs/extract_$st.log
done
echo ALL_DONE
