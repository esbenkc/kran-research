#!/bin/bash
# Serial schedule (room #3447 option A): when the (c) job has written all 5 own reads, stop it, run c2 (pilot + fits)
# alone, then the held-text reads, then the pooled fact-present fit. Run from kran-research/.
P=projects/megaprobe-spin-lies; L=$P/data/logs
until [ -f $P/data/fits/ens_s1_own.json ]; do sleep 15; done
pkill -f "\.venv/bin/python $P/probes.py fit" && echo "$(date '+%H:%M:%S') stopped (c) job after 5 own reads" >> $L/chain.log
for i in $(seq 1 40); do  # the old job's next step ('probes.py pooled') must not run next to c2
  pkill -f "probes.py pooled" && echo "$(date '+%H:%M:%S') stopped the old job's pooled step" >> $L/chain.log
  sleep 1
done
sleep 2
$P/watch_pattern.sh "megaprobe-spin-lies/fit_c2.py" 2300 $L/footprint_c2.log &
uv run python $P/fit_c2.py pilot > $L/c2_pilot.log 2>&1; echo "$(date '+%H:%M:%S') c2 pilot exit $?" >> $L/chain.log
uv run python $P/fit_c2.py fit base lam1_s0 lam1_s1 ens_s0 ens_s1 > $L/c2_fit.log 2>&1; echo "$(date '+%H:%M:%S') c2 fit exit $?" >> $L/chain.log
$P/watch_pattern.sh "megaprobe-spin-lies/probes.py" 3400 $L/footprint_held.log &
uv run python $P/probes.py fit base lam1_s0 lam1_s1 ens_s0 ens_s1 > $L/fit_held.log 2>&1; echo "$(date '+%H:%M:%S') held exit $?" >> $L/chain.log
uv run python $P/probes.py pooled > $L/pooled.log 2>&1; echo "$(date '+%H:%M:%S') pooled exit $?" >> $L/chain.log
wait
