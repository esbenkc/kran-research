#!/bin/bash
# Vera's own-engine check of build adapters, run only in Bruno's posted GPU gap (one job at a time).
cd "$(dirname "$0")"
PY=../../env/.venv/bin/python
R=../../kran-research/projects/multi-token-recompute/data/runs
for run in "$@"; do
  /usr/bin/time -l $PY vera_engine.py --n 64 --B 8 --seeds 2 --adapter $R/$run/adapter.pt --out vera_own_$run.npz 2>&1 \
    | grep -E "saved|maximum resident|peak memory|Error" 
  $PY vera_stats.py vera_own_$run.npz vera_own_$run.json 2>/dev/null >/dev/null
  $PY -c "import json;d=json.load(open('vera_own_$run.json'));print('$run',{k:d[k] for k in ['f','D_avg','pinsker','cheat_none_on_none_text','cheat_leaky_on_leaky_text','gap_full_pts','gap_text_pts','dec_abs_spill_A']})"
done
