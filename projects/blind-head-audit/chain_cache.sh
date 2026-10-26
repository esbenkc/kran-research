#!/bin/bash
# J1 (1.7B) then J2 (0.6B) neutral caches, serial, each under the 5.0 GB watchdog (Vera gate #7498).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1026-blind-head-audit/env/.venv/bin/python
for M in Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B; do
  T=$(basename $M)
  echo "START cache $M $(date)" >> logs/chain.log
  $PY watchdog.py --cap-gb 5.0 --log logs/cache_$T.log -- $PY cache.py --model $M
  echo "END cache $M rc=$? $(date)" >> logs/chain.log
done
echo "CACHE_DONE $(date)" >> logs/chain.log
