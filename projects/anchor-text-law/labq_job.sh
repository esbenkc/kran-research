#!/bin/bash
# One laptop-queue job (room defense-1123-anchor-text-law). Runs in projects/anchor-text-law of labq's clean clone.
#   bash labq_job.sh <agent slot> <model> <stop-at ISO> <n-main>
#   J1: bash labq_job.sh A gpt-oss-20b 2026-11-23T03:05:00-08:00 60
#   J2: bash labq_job.sh B qwen3-8b-nothink 2026-11-23T03:05:00-08:00 42
# Steps: move to the branch head (a resumed job skips every run an earlier job pushed); refuse to run if the harness
# differs from the frozen code commit; gate block 0 at step cap 16; web/gate_decide.py (model swap for B, step cap);
# then every cell (gate0 skipped if done, main k-major, gate block 1, filler) until done or --stop-at.
# Outputs: web/out/<slot>/<model>/ only (box-unique). PY=echo = dry run.
# The whole body is one function, so bash has read all of it before `git reset` can change this file.
main() {
  local CODE_SHA=caaf5c4e4 BR=swarm/defense-1123-anchor-text-law SLOT=$1 MODEL=$2 STOP=$3 NMAIN=$4
  local PY=${PY:-uv run python} TAG="$1/$2" LINE NEWMODEL NEWTAG STEPS ACT
  git fetch -q origin "$BR" && git reset -q --hard FETCH_HEAD || { echo "labq_job: fetch/reset failed"; return 5; }
  echo "labq_job: resume base $(git rev-parse HEAD)"
  if ! git diff --quiet "$CODE_SHA" HEAD -- web ':(exclude)web/out' pyproject.toml uv.lock; then
    echo "labq_job: harness drift vs $CODE_SHA"; git diff --stat "$CODE_SHA" HEAD -- web ':(exclude)web/out' pyproject.toml uv.lock
    return 4
  fi
  uv sync --frozen || return 6
  export CANARY_PORT=8823
  for round in 1 2; do
    CANARY_MAX_STEPS=16 caffeinate -i $PY web/run.py --model "$MODEL" --tag "$TAG" --phase gate0 --stop-at "$STOP" || return 7
    if [ -n "${DRY:-}" ]; then LINE="$MODEL $TAG 16 keep"; else LINE=$(uv run python web/gate_decide.py --tag "$TAG" --model "$MODEL") || return 8; fi
    echo "labq_job: gate decision: $LINE"
    read -r NEWMODEL NEWTAG STEPS ACT <<< "$LINE"
    [ "$ACT" = "swap" ] || break
    MODEL=$NEWMODEL TAG=$NEWTAG
  done
  CANARY_MAX_STEPS=$STEPS caffeinate -i $PY web/run.py --model "$MODEL" --tag "$TAG" --phase all --stop-at "$STOP" --n-main "$NMAIN"
}
main "$@"
exit $?
