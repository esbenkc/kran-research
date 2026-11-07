#!/bin/bash
# Vera: independent forward (own engine) of every saved run adapter under its own clamp; compares to build's final rows.
cd /Users/esben/work/theory-1107-near-tie-clamp
PY=env/.venv/bin/python; O=swarm-theory-1107-near-tie-clamp/skeptic/out; D=kran-research/projects/head-clamp-retrain/data
for run in "$@"; do
  heads=$(python3 -c "import json;print(','.join(json.load(open('$D/runs/$run.json'))['heads']))")
  [ -f $O/fwd_$run.json ] || $PY swarm-theory-1107-near-tie-clamp/skeptic/vera_tail.py $D/adapters/${run}_adapter.pt --heads "$heads" --out $O/fwd_$run.json 2>&1 | grep -E "^\{|Error" | cut -c1-260
  python3 swarm-theory-1107-near-tie-clamp/skeptic/vera_cmp.py $O/fwd_$run.json $D/runs/$run.json final/test_merged_clamp
done
