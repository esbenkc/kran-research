#!/bin/bash
# Glue (not part of the method): end the pilot at Felix's 1e6 cap, write pilot.json, wait for the PREREG
# freeze commit (Tove condition a, #2715), then start the 12 scored runs.
cd "$(dirname "$0")"
LOG=data/runs/run_scored.log
mkdir -p data/runs
while [ ! -f data/pilot/pilot.json ] && [ ! -f data/pilot/lam1e+06__seed99.npz ]; do sleep 10; done
pkill -f "pilot.py" ; pkill -f "attack.py --lam 3162" ; sleep 3
uv run python pilot.py 2>&1 | grep -E "^\{|frozen" >> "$LOG"
echo "$(date '+%H:%M:%S') pilot.json written; waiting for the PREREG.md freeze commit" >> "$LOG"
until git -C .. log --oneline -- rethreshold-probe-evasion/PREREG.md | grep -q .; do sleep 10; done
echo "$(date '+%H:%M:%S') PREREG commit: $(git -C .. log -1 --format='%h %ci' -- rethreshold-probe-evasion/PREREG.md)" >> "$LOG"
./run_scored.sh
