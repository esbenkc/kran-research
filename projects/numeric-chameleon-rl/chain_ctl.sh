#!/bin/bash
# 13:35 re-plan (Bruno): J3 of the status cells starts as soon as r50kl s0 (16384) ends; TAGkl and r100 move to
# lane D after SHUFkl. Lane C' = wait for watchdog pid $1; lane D' = wait for watchdog pid $2, then its queue.
cd "$(dirname "$0")"
( while kill -0 $1 2>/dev/null; do sleep 5; done; echo "C' done $(date +%T)"; touch /tmp/nc_lane_c_done ) &
( while kill -0 $2 2>/dev/null; do sleep 5; done; echo "D' start $(date +%T)"
  STEPS=8192 ./chain_j2.sh "shuf 0 1e-4 --kl 1 --save-at 4096"
  STEPS=4096 ./chain_j2.sh "tag 0 1e-4 --kl 1" "r100 0 1e-4"
  echo "D' done $(date +%T)"; touch /tmp/nc_lane_d_done ) &
wait; echo CTLDONE $(date +%T)
