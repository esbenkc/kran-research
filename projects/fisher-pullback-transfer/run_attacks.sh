#!/bin/bash
# Run order (DEVIATIONS 7ca35fe): worker A = seed 0 of all targets, then lambda=0 runs of targets 1-3, then seed 2 of targets 1-3;
# worker B = seed 1 of all targets, then lambda=0 runs of targets 4-5, then seed 2 of targets 4-5. 2 workers max (Vera's gate).
cd "$(dirname "$0")"
T=("dbpedia:NaturalPlace" "emotion:fear" "lang:it" "sst2:positive" "yahoo:Education & Reference")
run() { uv run python attack.py run "$@" 2>&1 | grep --line-buffered '^{'; }
if [ "$1" = A ]; then
  for t in "${T[@]}"; do run --target "$t" --seed 0; done
  for t in "${T[@]:0:3}"; do run --target "$t" --seed 0 --lam 0; done
  for t in "${T[@]:0:3}"; do run --target "$t" --seed 2; done
else
  for t in "${T[@]}"; do run --target "$t" --seed 1; done
  for t in "${T[@]:3:2}"; do run --target "$t" --seed 0 --lam 0; done
  for t in "${T[@]:3:2}"; do run --target "$t" --seed 2; done
fi
