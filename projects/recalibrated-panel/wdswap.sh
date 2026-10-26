#!/bin/bash
# wdswap.sh: for every running attack_recal.py, replace its 9000 MB watchdog by a 9216 MiB (= 9 GiB) one, same log.
# Runs until the chain ends. Extract jobs keep the 9000 cap.
cd "$(dirname "$0")"
while pgrep -f "run_chain.sh|chain_wait.sh" >/dev/null; do
  for apid in $(pgrep -f "attack_recal.py --lam"); do
    w=$(pgrep -f "watchdog.sh $apid 9000")
    if [ -n "$w" ]; then
      log=$(ps -o command= -p $w | awk '{print $NF}')
      kill $w && nohup ./watchdog.sh $apid 9216 $log >/dev/null 2>&1 &
      echo "$(date +%H:%M:%S) swapped watchdog for $apid -> 9216 ($log)" >> data/logs/wdswap.log
    fi
  done
  sleep 10
done
