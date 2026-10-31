#!/bin/bash
# judge every new TEST-harmful generation file (D11); cached, so re-runs cost nothing. Stops at 22:15 PDT or on STOP file.
cd /Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/skeptic
while [ ! -f own/JUDGE_STOP ] && [ "$(date +%H%M)" -lt 2215 ]; do
  new=""
  for f in ../data/runs/test_harmful/*.jsonl; do
    k="test_harmful/$(basename $f .jsonl)"
    # only judge complete files (200 lines) not yet labelled
    if [ "$(wc -l < $f)" -ge 200 ] && ! grep -q "\"$k\"" vera_judge_test.json; then new="$new $f"; fi
  done
  if [ -n "$new" ]; then uv run --no-project --with anthropic python vera_judge.py vera_judge_test.json $new >> own/judge_watch.log 2>&1; fi
  sleep 60
done
