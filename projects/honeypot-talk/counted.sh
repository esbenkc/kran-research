#!/bin/bash
# Counted runs (PREREG v1.3 s4 + D3): main = seeds 0-29 all four cells + prompt-on seeds 30-49 (800 runs),
# then D3: prompt-on seeds 50-69 (both decoys, all five models) only if logged spend after the main run is <= $3.70.
cd /Users/esben/work/defense-1031-say-do-trap-gap/kran-research/projects/honeypot-talk
PY=/Users/esben/work/defense-1031-say-do-trap-gap/env/.venv/bin/python
with-model-keys $PY run.py counted --seeds 50 --off-seeds 30 --workers 12 --budget 4.50 > logs/counted.log 2>&1
SPENT=$($PY -c "import run; print(run.spent())")
echo "spent after main: $SPENT" >> logs/counted.log
if $PY -c "import sys, run; sys.exit(0 if run.spent() <= 3.70 else 1)"; then
  echo "D3: spend <= 3.70, running prompt-on seeds 50-69" >> logs/counted.log
  with-model-keys $PY run.py counted --seeds 70 --off-seeds 30 --workers 12 --budget 4.50 > logs/counted_d3.log 2>&1
else
  echo "D3: spend > 3.70, extension not run" >> logs/counted.log
fi
echo COUNTED_END >> logs/counted.log
