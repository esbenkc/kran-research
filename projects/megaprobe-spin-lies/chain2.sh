#!/bin/bash
# Second half of the serial schedule (after all 5 own reads): c2 pilot + fits alone, then held text, then pooled.
# The first chain's 2.3 GB c2 watcher killed the c2 pilot at 2.4 GB (03:24:52); c2 line is now 3.3 GB. Run from kran-research/.
P=projects/megaprobe-spin-lies; L=$P/data/logs
$P/watch_pattern.sh "megaprobe-spin-lies/fit_c2.py" 3300 $L/footprint_c2b.log &
uv run python $P/fit_c2.py pilot > $L/c2_pilot.log 2>&1; echo "$(date '+%H:%M:%S') c2 pilot exit $?" >> $L/chain.log
[ -f $P/data/hp_c2.json ] || { echo "$(date '+%H:%M:%S') no hp_c2.json, stop" >> $L/chain.log; exit 1; }
uv run python $P/fit_c2.py fit base lam1_s0 lam1_s1 ens_s0 ens_s1 > $L/c2_fit.log 2>&1; echo "$(date '+%H:%M:%S') c2 fit exit $?" >> $L/chain.log
$P/watch_pattern.sh "megaprobe-spin-lies/probes.py" 4900 $L/footprint_held.log &
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12 uv run python $P/probes.py fit lam1_s0 lam1_s1 ens_s0 ens_s1 > $L/fit_held.log 2>&1; echo "$(date '+%H:%M:%S') held exit $?" >> $L/chain.log
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.12 uv run python $P/probes.py pooled > $L/pooled.log 2>&1; echo "$(date '+%H:%M:%S') pooled exit $?" >> $L/chain.log
wait
