#!/bin/bash
# Vera: rebuild every counted D8 model (step 12000) at h_2 (vr_) and h_4 (vr4_) that is not rebuilt yet.
cd "$(dirname "$0")"
ROOT=../../kran-research/projects/gated-sinks
for L in 2 4; do
  pre=vr; [ $L = 4 ] && pre=vr4
  for f in $ROOT/runs/*/measure_L${L}_m256_k8_c12000.json; do
    [ -f "$f" ] || continue
    run=$(basename $(dirname $f))
    [ -f ${pre}_$run.json ] && continue
    uv run --no-project --with numpy --with torch --with scipy python vera_rebuild.py --root $ROOT --runs $run --tag L${L}_m256_k8_c12000 --out ${pre}_$run.json > /dev/null 2>&1 && echo "rebuilt $pre $run"
  done
done
ls vr_*.json vr4_*.json 2>/dev/null | wc -l
