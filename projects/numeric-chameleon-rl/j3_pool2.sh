#!/bin/bash
# J3 resume after Vera's rebuild slot: worker 1 starts when vera_queue.log has 'end'; worker 2 when lane D is done too.
# A run whose merged weights do not exist yet goes back to the end of the queue (SHUFkl, TAGkl, r100 finish later).
cd "$(dirname "$0")"
Q=logs/j3_queue_rest.txt
VLOG=../../../swarm-theory-1025-numeric-chameleon/skeptic/vera_queue.log
pop() {
  until mkdir logs/.j3lock 2>/dev/null; do sleep 1; done
  n=$(head -1 $Q); [ -n "$n" ] && tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q
  rmdir logs/.j3lock; echo "$n"
}
push() { until mkdir logs/.j3lock 2>/dev/null; do sleep 1; done; echo "$1" >> $Q; rmdir logs/.j3lock; }
worker() {
  while true; do
    n=$(pop); [ -z "$n" ] && break
    if [ "$n" != base ] && [ ! -f raw/runs/${n}_merged.pt ]; then
      left=$(wc -l < $Q); push "$n"; [ "$left" -eq 0 ] && sleep 30; continue
    fi
    GEN=0
    case "$n" in
      base|sft_*|r0_*|tag_*|shuf_s0_lr0.0001_kl1) GEN=8 ;;
    esac
    GEN=$GEN ./chain_j3.sh "$n" 2>&1 | grep -E "J3DONE|FAIL"
  done
  echo "worker $1 done $(date +%T)"
}
until grep -q '^end' $VLOG 2>/dev/null; do sleep 10; done
echo "worker 1 start $(date +%T)"; worker 1 &
until [ -f /tmp/nc_lane_d_done ]; do sleep 10; done
echo "worker 2 start $(date +%T)"; worker 2 &
wait; echo "J3POOL2DONE $(date +%T)"; touch /tmp/drew_slot
