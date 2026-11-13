#!/bin/sh
# Post-run chain v2 (replaces post.sh's order; Felix #21425/#21428, Tove #21429, Vera #21427, Kasper #21431; D4).
# Serial, one heavy job at a time, each step continues on error. Waits until live.py (live_k4) has exited; E = last read.
#   1. served planted read on adapter_E (status cell)
#   2. fresh-sample trace read: pool from adapter_E, 200 TEST prompts x 4 draws (PREREG fallback size, clock), + reads 0/E
#   3. diagnostic: cached.py on the d010_s0 pools/seed WITH live.py's thought KL (0.1, 16 rows x4), --stop-at 03:52 PST;
#      if it dies within 150 s, the cached seed-1 cell (on_d0.10_s1, parent recipe) runs instead, --stop-at 03:55
#   4. thought drift (live E, cached 1,200)
#   5. control word ' Let': only if it can start before 03:56 PST
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
while pgrep -f "live.py --k-stale 4" >/dev/null; do sleep 5; done
E=$(/usr/bin/python3 -c "import json;print(json.loads(open('data/runs/live_k4/reads.jsonl').read().splitlines()[-1])['step'])")
A=data/runs/live_k4/adapter_$E.pt
mkdir -p data/reads
echo "post2 start E=$E $(date '+%F %T')"
$PY watchdog.py --cap-gb 8.5 --log logs/served_$E.log -- $PY served.py --policy $A --R 2 --out data/reads/served_live_$E.npz \
  || echo "served FAILED"
echo "fresh start $(date '+%T')"
$PY watchdog.py --cap-gb 8.5 --log logs/fresh_pool.log -- $PY pool.py --mode test --policy $A --R 4 --X 1205 --B 64 --seed 0 \
     --out data/pool/fresh$E \
  && $PY watchdog.py --cap-gb 8.5 --log logs/fresh_read.log -- $PY readpool.py --pool data/pool/fresh$E --out data/reads/fresh$E \
     --adapters 0=data/runs/live_k4/adapter_0.pt,$E=$A \
  || echo "fresh FAILED"
echo "diag start $(date '+%T')"
T0=$(date +%s)
$PY watchdog.py --cap-gb 8.5 --log logs/cachedkl_s0.log -- $PY cached.py --name cachedkl_s0 --test data/pool/test \
     --off data/pool/off --on data/pool/on_d0.10 --seed 0 --steps 1500 --no-std --lam 0.42 --read-early 100 --read-every 100 \
     --kl-prefix 0.1 --kl-rows 16 --kl-chunk 4 --stop-at 03:52 \
  || echo "cachedkl FAILED"
if [ $(( $(date +%s) - T0 )) -lt 150 ]; then
  echo "cachedkl died early -> seed-1 fallback $(date '+%T')"
  $PY watchdog.py --cap-gb 8.5 --log logs/cached_s1.log -- $PY cached.py --name cached_s1 --test data/pool/test \
       --off data/pool/off --on data/pool/on_d0.10_s1 --seed 1 --steps 1500 --no-std --lam 0.42 --read-early 100 \
       --read-every 100 --stop-at 03:55 || echo "cached_s1 FAILED"
fi
$PY watchdog.py --cap-gb 8.5 --log logs/drift_live.log -- $PY drift.py --end $A --n 256 --out data/reads/drift_live_$E.json \
  || echo "drift live FAILED"
$PY watchdog.py --cap-gb 8.5 --log logs/drift_cached.log -- $PY drift.py --end data/runs/cached_d010_s0/adapter_1200.pt --n 256 \
  --out data/reads/drift_cached_1200.json || echo "drift cached FAILED"
if [ "$(date +%H%M)" -lt 0356 ]; then
  $PY watchdog.py --cap-gb 8.5 --log logs/ctrl_pool.log -- $PY ctrl_pool.py --src data/pool/test --X 6771 --out data/pool/test_let \
    && $PY watchdog.py --cap-gb 8.5 --log logs/ctrl_read.log -- $PY readpool.py --pool data/pool/test_let --out data/reads/ctrl_let \
       --adapters 0=data/runs/live_k4/adapter_0.pt,600=data/runs/live_k4/adapter_600.pt,$E=$A \
    || echo "ctrl FAILED"
else
  echo "ctrl skipped (clock) $(date '+%T')"
fi
echo "post2 done $(date '+%F %T')"
