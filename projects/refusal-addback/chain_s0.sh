#!/bin/sh
# Seed-0 pipeline, strictly serial (one model process at a time; DEVIATIONS D4: footprint <= 5 GB).
# Each step is resume-safe. Logs in logs/. Peak footprint per step in logs/footprint.log.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
step() {
  name=$1; shift
  echo "=== $name $(date +%H:%M:%S)" >> logs/chain.log
  $PY "$@" > logs/$name.log 2>&1 &
  pid=$!
  peak=0; rpeak=0
  while kill -0 $pid 2>/dev/null; do
    fp=$(footprint $pid 2>/dev/null | awk '/phys_footprint:/{v=$2; u=$3; if(u=="KB")v=v/1024; if(u=="GB")v=v*1024; if(u=="B")v=0; printf "%d", v; exit}')
    if [ -n "$fp" ] && [ "$fp" -gt "$peak" ]; then peak=$fp; fi
    rss=$(ps -o rss= -p $pid 2>/dev/null | awk '{printf "%d", $1/1024}')
    if [ -n "$rss" ] && [ "$rss" -gt "$rpeak" ]; then rpeak=$rss; fi
    if [ -n "$fp" ] && [ "$fp" -gt 6000 ]; then echo "KILL $name footprint $fp MB" >> logs/chain.log; kill $pid; fi
    sleep 5
  done
  wait $pid; rc=$?
  echo "$name peak_footprint_MB=$peak peak_rss_MB=$rpeak rc=$rc $(date +%H:%M:%S)" >> logs/footprint.log
  [ $rc -eq 0 ] || { echo "FAIL $name rc=$rc" >> logs/chain.log; exit 1; }
}
[ -f runs/dirs.npy ] || step direction addback.py direction
[ -f runs/base/readout.json ] || step readout_base addback.py readout --model base
step test_base_none addback.py test --model base
[ -f runs/strip_s0/adapter/adapter_config.json ] || step strip_s0 strip.py --seed 0
[ -f runs/strip_s0/readout.json ] || step readout_s0 addback.py readout --model strip_s0
step test_s0_none addback.py test --model strip_s0
step dev_s0      addback.py dev --model strip_s0
true # dev_base runs in chain_test.sh after TEST            
echo "=== chain_s0 done $(date +%H:%M:%S)" >> logs/chain.log
