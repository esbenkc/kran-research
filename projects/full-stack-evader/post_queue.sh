#!/bin/bash
# After the scored queue: refits of every attacked state (CPU), then analysis.py, then export.py. Serial, light.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
for st in a__gap__lam10000__seed0 a__gap__lam10000__seed1 a__gap__lam10000__seed2 b__gap300__lam10000__seed0 \
          b__gap300__lam10000__seed1 b__gap300__lam10000__seed2 a__gap__lam0__seed0 b__gap__lam10000__seed0 \
          a__softplus__lam10000__seed0; do
  [ -f data/refit/$st.json ] || $PY refit.py $st > data/logs/refit_$st.log 2>&1 || echo "FAILED refit $st"
done
$PY analysis.py > data/logs/analysis.log 2>&1 || echo "FAILED analysis"
$PY export.py > data/logs/export.log 2>&1 || echo "FAILED export"
echo "$(date +%H:%M:%S) POST_DONE"
