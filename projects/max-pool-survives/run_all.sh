#!/bin/bash
# J1-J4 of max-pool-survives, serial (one heavy job at a time). Logs: data/logs/<job>.log
set -e
cd "$(dirname "$0")"
mkdir -p data/logs
[ -f data/clean_tokens.npz ] || uv run python clean_tokens.py 2>&1 | tee data/logs/clean_tokens.log
for s in 0 1 2; do
  [ -f data/runs/lam0__seed${s}__tokens.npz ] || uv run python attack.py --lam 0 --seed $s 2>&1 | tee data/logs/lam0__seed${s}.log
done
echo ALL_DONE
