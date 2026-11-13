#!/bin/sh
# Post-run chain (DEVIATIONS D5: Tove #21434/#21437, Kasper #21436, Felix #21438). Serial, one heavy job at a time,
# each step continues on error. Waits until live.py (live_k4) has exited; E = last read.
#   1. served planted read on adapter_E (status cell)
#   2. fresh-sample trace read: pool from adapter_E, 200 TEST prompts x 8 draws (stored pool's noise seeds), reads 0/E
#   3. cell A (exploratory): cached.py on the d010_s0 pools/seed/recipe WITH live.py's thought KL (0.1, 16 rows x4),
#      reads every 100 from 300, --stop-at 03:55 PST; gate: exit 3 if updates 6-15 > 3.5 s/update, or any exit within
#      150 s -> cell B (cached seed 1, on_d0.10_s1, parent recipe) instead, --stop-at 03:55
#   ' Let' control and drift reads: dropped (D5).
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
$PY watchdog.py --cap-gb 8.5 --log logs/fresh_pool.log -- $PY pool.py --mode test --policy $A --R 8 --X 1205 --B 64 --seed 0 \
     --out data/pool/fresh$E \
  && $PY watchdog.py --cap-gb 8.5 --log logs/fresh_read.log -- $PY readpool.py --pool data/pool/fresh$E --out data/reads/fresh$E \
     --adapters 0=data/runs/live_k4/adapter_0.pt,$E=$A \
  || echo "fresh FAILED"
echo "cell A start $(date '+%T')"
T0=$(date +%s)
$PY watchdog.py --cap-gb 8.5 --log logs/cachedkl_s0.log -- $PY cached.py --name cachedkl_s0 --test data/pool/test \
     --off data/pool/off --on data/pool/on_d0.10 --seed 0 --steps 1500 --no-std --lam 0.42 --read-early 100 --read-every 100 \
     --first-read 300 --no-read0 --max-s-upd 3.5 --kl-prefix 0.1 --kl-rows 16 --kl-chunk 4 --stop-at 03:55
RC=$?
echo "cell A exit $RC after $(( $(date +%s) - T0 )) s $(date '+%T')"
if [ $RC -eq 3 ] || [ $(( $(date +%s) - T0 )) -lt 150 ]; then
  echo "cell A gated -> cell B (seed 1) $(date '+%T')"
  $PY watchdog.py --cap-gb 8.5 --log logs/cached_s1.log -- $PY cached.py --name cached_s1 --test data/pool/test \
       --off data/pool/off --on data/pool/on_d0.10_s1 --seed 1 --steps 1500 --no-std --lam 0.42 --read-early 100 \
       --read-every 100 --first-read 300 --stop-at 03:55 || echo "cell B FAILED"
fi
echo "post2 done $(date '+%F %T')"
