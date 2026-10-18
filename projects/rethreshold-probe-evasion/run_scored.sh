#!/bin/bash
# The 12 scored runs, strictly serial (compute gate #2731). Grid from data/pilot/pilot.json (frozen_grid).
# Order: largest lambda first (the headline rule reads the largest qualifying lambda), lambda = 0 last.
# Each run is skipped if its .npz already exists. Log: data/runs/run_scored.log
cd "$(dirname "$0")"
mkdir -p data/runs
GRID=$(uv run python -c "import json; g=json.load(open('data/pilot/pilot.json'))['frozen_grid']; print(' '.join(str(x) for x in sorted(g, reverse=True)))")
for lam in $GRID; do
  for seed in 0 1 2; do
    stem=$(uv run python -c "print(f'lam{float($lam):g}__seed$seed')")
    if [ -f "data/runs/$stem.npz" ]; then echo "skip $stem" >> data/runs/run_scored.log; continue; fi
    echo "$(date '+%H:%M:%S') start $stem" >> data/runs/run_scored.log
    uv run python attack.py --lam "$lam" --seed "$seed" 2>&1 | grep -E '^\{"lam"|Error|error' >> data/runs/run_scored.log
    echo "$(date '+%H:%M:%S') end $stem" >> data/runs/run_scored.log
  done
done
echo "$(date '+%H:%M:%S') ALL DONE" >> data/runs/run_scored.log
