#!/bin/bash
# D3 queue (Felix #6956). Waits for the group-flag R50 pair to exit, gives Iris a slot (/tmp/iris_slot ->
# she touches /tmp/iris_done, or 8 min pass), then runs lanes C and D in parallel (2 J2 jobs, each <= 2.4 GB).
cd "$(dirname "$0")"
# (restarted 12:52: the group-flag pair and the smoke are done)
for i in $(seq 48); do [ -f /tmp/iris_done ] && break; sleep 5; done
echo "lanes start $(date +%T)"
STEPS=8192 ./chain_j2.sh "r50 0 1e-4 --mix --save-at 4096" > logs/laneC1.out 2>&1 &
C=$!
STEPS=8192 ./chain_j2.sh "r50 1 1e-4 --mix --save-at 4096" "r50 1 1e-4 --kl 1 --save-at 4096" \
  "shuf 0 1e-4 --kl 1 --save-at 4096" > logs/laneD1.out 2>&1 &
D=$!
wait $C; echo "C1 done $(date +%T)"
STEPS=16384 ./chain_j2.sh "r50 0 1e-4 --kl 1 --save-at 4096,8192" > logs/laneC2.out 2>&1 &
C=$!
wait $D; echo "D1 done $(date +%T)"
STEPS=4096 ./chain_j2.sh "tag 0 1e-4 --kl 1" "r100 0 1e-4" > logs/laneD2.out 2>&1
echo "D2 done $(date +%T)"; touch /tmp/nc_lane_d_done
wait $C; echo "C2 done $(date +%T)"; touch /tmp/nc_lane_c_done
echo D3DONE
