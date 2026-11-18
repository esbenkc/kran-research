#!/bin/sh
# Update-1 sampling only (DEVCHK_SAMPLE_ONLY in this scratch live.py), seed 0, vs the mini's stored kl0_s0.
# Arg 1: variant  A = no MPS watermark cap, --time-only (no pool read 0)
#                 B = no MPS watermark cap, full mode (pool read 0 first, as the stored run did)
# Watchdog kills at 5.0 GB phys footprint (compute gate Vera #25302).
set -u
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1113-on-policy-planted/env/.venv/bin/python
V=$1; S=${2:-0}
FLAGS="--kl-prefix 0 --steps 1200 --P 8 --G 8 --lr 3e-4 --kl 0.1 --kl-rows 16 --lam 0.42 --sd-ref 0.2 --r-solve 0.6
 --plant-tok 1205 --cue-delta 0.10 --dec-bias -2.255 --k-stale 4 --chunk 4 --read-every 100 --stop-onset 0.3 --stop-confirm 2"
N=dev${V}_s$S
rm -rf data/runs/$N
mkdir -p logs
if [ "$V" = A ]; then EXTRA="--time-only 50"; else EXTRA=""; fi
echo "start $N $(date '+%F %T')"
DEVCHK_SAMPLE_ONLY=1 $PY watchdog.py --cap-gb 5.0 --log logs/$N.log -- $PY live.py --name $N --seed $S $EXTRA $FLAGS
echo "exit $? $N $(date '+%F %T')"; tail -2 logs/$N.log
