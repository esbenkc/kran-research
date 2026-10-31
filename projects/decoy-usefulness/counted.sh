#!/bin/bash
# T4 counted runs (PREREG s4, s11): strong five 20/cell, weak three 10/cell, then the variant cell if spend < $3.80.
cd /Users/esben/work/defense-1031-usefulness-beats/kran-research/projects/decoy-usefulness
PY=/Users/esben/work/defense-1031-usefulness-beats/env/.venv/bin/python
with-model-keys $PY run.py counted --seeds 20 --weak-seeds 10 --workers 12 --budget 4.50 > logs/counted.log 2>&1
SPENT=$($PY -c "import run; print(run.spent())")
echo "spent after main: $SPENT" >> logs/counted.log
if $PY -c "import sys, run; sys.exit(0 if run.spent() < 3.80 else 1)"; then
  with-model-keys $PY run.py counted --seeds 20 --workers 12 --budget 3.80 --arms useless_real_bucket --models gemini-3.1-flash-lite gemini-3.5-flash-lite gpt-5.4-nano gpt-5.4-mini claude-haiku-4-5-20251001 > logs/counted_variant.log 2>&1
fi
echo COUNTED_END >> logs/counted.log
