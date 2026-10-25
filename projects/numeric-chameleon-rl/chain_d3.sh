#!/bin/bash
# D3 queue (Felix #6956, Bruno #6961, Tove #6962). Waits for Iris's slot (/tmp/iris_done, max 4 min), then two lanes
# (2 J2 jobs, each <= 2.4 GB). r50mix runs stop by themselves once absorbed (rl.py, Tove #6962).
cd "$(dirname "$0")"
for i in $(seq 48); do [ -f /tmp/iris_done ] && break; sleep 5; done
echo "lanes start $(date +%T)"
( STEPS=8192 ./chain_j2.sh "r50 0 1e-4 --mix --save-at 4096"
  STEPS=16384 ./chain_j2.sh "r50 0 1e-4 --kl 1 --save-at 4096,8192"
  STEPS=4096 ./chain_j2.sh "tag 0 1e-4 --kl 1" "r100 0 1e-4"
  echo "C done $(date +%T)"; touch /tmp/nc_lane_c_done ) > logs/laneC.out 2>&1 &
( STEPS=8192 ./chain_j2.sh "r50 1 1e-4 --mix --save-at 4096"
  STEPS=16384 ./chain_j2.sh "r50 1 1e-4 --kl 1 --save-at 4096,8192"
  STEPS=8192 ./chain_j2.sh "shuf 0 1e-4 --kl 1 --save-at 4096"
  echo "D done $(date +%T)"; touch /tmp/nc_lane_d_done ) > logs/laneD.out 2>&1 &
wait
echo D3DONE $(date +%T)
