#!/bin/bash
# Two J3 workers fed from one queue (logs/j3_queue.txt, one run name per line, status cells first).
# Worker 1 starts when the first J2 lane ends, worker 2 when the second ends: never more than 2 heavy jobs.
cd "$(dirname "$0")"
Q=logs/j3_queue.txt
pop() {  # atomic pop with a mkdir lock
  until mkdir logs/.j3lock 2>/dev/null; do sleep 1; done
  n=$(head -1 $Q); [ -n "$n" ] && tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q
  rmdir logs/.j3lock; echo "$n"
}
worker() {
  while true; do
    n=$(pop); [ -z "$n" ] && break
    if [ ! -f raw/runs/${n}_merged.pt ] && [ "$n" != base ]; then echo "skip $n (no merged weights yet) $(date +%T)"; continue; fi
    GEN=0
    case "$n" in
      base|sft_*|r0_*|r50_s?_lr0.0001_kl1_at8192|tag_*|shuf_*) GEN=8 ;;
    esac
    GEN=$GEN ./chain_j3.sh "$n" 2>&1 | grep -E "J3DONE|FAIL"
  done
  echo "worker $1 done $(date +%T)"
}
until [ -f /tmp/nc_lane_c_done ] || [ -f /tmp/nc_lane_d_done ]; do sleep 10; done
echo "worker 1 start $(date +%T)"; worker 1 &
until [ -f /tmp/nc_lane_c_done ] && [ -f /tmp/nc_lane_d_done ]; do sleep 10; done
echo "worker 2 start $(date +%T)"; worker 2 &
wait; echo "J3POOLDONE $(date +%T)"  # drew_slot moved to j3_pool2.sh
