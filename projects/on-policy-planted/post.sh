#!/bin/sh
# Post-run reads (PREREG v2 Reads), serial, one heavy job at a time, each step continues on error.
# Waits until live.py (live_k4) has exited, then on the LAST checkpoint E (1,200 unless the clock stopped the run):
#   1. served planted read: E adapter, 200 TEST prompts x 2 fresh draws, X forced vs banned at token 0 (CRN)
#   2. control-token twin pool (X' = 6771 ' Let') + reads of adapters 0, 600, E on it
#   3. fresh-sample pool from the E adapter (200 TEST prompts x R draws, same noise seeds as the stored pool) + reads of
#      adapters 0 and E on it (fresh D_R, y_R, s_R, and the fresh planted switch)
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
R=${R:-8}
while pgrep -f "live.py --k-stale 4" >/dev/null; do sleep 10; done
E=$(/usr/bin/python3 -c "import json;print(json.loads(open('data/runs/live_k4/reads.jsonl').read().splitlines()[-1])['step'])")
A=data/runs/live_k4/adapter_$E.pt
mkdir -p data/reads
echo "post start E=$E $(date '+%F %T')"
$PY watchdog.py --cap-gb 8.5 --log logs/served_$E.log -- $PY served.py --policy $A --R 2 --out data/reads/served_live_$E.npz \
  || echo "served FAILED"
$PY watchdog.py --cap-gb 8.5 --log logs/ctrl_pool.log -- $PY ctrl_pool.py --src data/pool/test --X 6771 --out data/pool/test_let \
  && $PY watchdog.py --cap-gb 8.5 --log logs/ctrl_read.log -- $PY readpool.py --pool data/pool/test_let --out data/reads/ctrl_let \
     --adapters 0=data/runs/live_k4/adapter_0.pt,600=data/runs/live_k4/adapter_600.pt,$E=$A \
  || echo "ctrl FAILED"
$PY watchdog.py --cap-gb 8.5 --log logs/fresh_pool.log -- $PY pool.py --mode test --policy $A --R $R --X 1205 --B 64 --seed 0 \
     --out data/pool/fresh$E \
  && $PY watchdog.py --cap-gb 8.5 --log logs/fresh_read.log -- $PY readpool.py --pool data/pool/fresh$E --out data/reads/fresh$E \
     --adapters 0=data/runs/live_k4/adapter_0.pt,$E=$A \
  || echo "fresh FAILED"
# 4. thought drift (per-token TV start vs adapter, on 256 stored test thoughts): live E vs the cached run's 1,200 adapter
$PY watchdog.py --cap-gb 8.5 --log logs/drift_live.log -- $PY drift.py --end $A --n 256 --out data/reads/drift_live_$E.json \
  || echo "drift live FAILED"
$PY watchdog.py --cap-gb 8.5 --log logs/drift_cached.log -- $PY drift.py --end data/runs/cached_d010_s0/adapter_1200.pt --n 256 \
  --out data/reads/drift_cached_1200.json || echo "drift cached FAILED"
echo "post done $(date '+%F %T')"
