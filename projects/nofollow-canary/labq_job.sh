#!/bin/bash
# One laptop-queue job (room defense-1124-length-or-markup). Runs in projects/nofollow-canary of labq's clean clone.
#   bash labq_job.sh <stop-at ISO> <n-main>
#   J1: bash labq_job.sh 2026-11-24T12:00:00-08:00 90
# Agent: Qwen3-8B 4-bit MLX, thinking ON (anchor-text-law's gate-fixed setting; no swap in this run), step cap 16.
# Order (web/run.py priority_list): 15 skill-gate runs (md view) first, then run i of every arm (90 each, mdpad8 60).
# Steps: move to the branch head (a resumed job skips every run an earlier job pushed); refuse to run if the harness
# differs from the frozen code commit; then every run until done or --stop-at. Outputs: web/out/qwen3-8b/ only.
# PY=echo = dry run. The whole body is one function, so bash has read all of it before `git reset` can change this file.
main() {
  local CODE_SHA=922a47530 BR=swarm/defense-1124-length-or-markup STOP=$1 NMAIN=$2
  local PY=${PY:-uv run python}
  git fetch -q origin "$BR" && git reset -q --hard FETCH_HEAD || { echo "labq_job: fetch/reset failed"; return 5; }
  echo "labq_job: resume base $(git rev-parse HEAD)"
  if ! git diff --quiet "$CODE_SHA" HEAD -- web ':(exclude)web/out' padding.txt pyproject.toml uv.lock; then
    echo "labq_job: harness drift vs $CODE_SHA"; git diff --stat "$CODE_SHA" HEAD -- web ':(exclude)web/out' padding.txt pyproject.toml uv.lock
    return 4
  fi
  uv sync --frozen || return 6
  export CANARY_PORT=8824 CANARY_MAX_STEPS=16
  caffeinate -i $PY web/run.py --model qwen3-8b --tag qwen3-8b --phase all --stop-at "$STOP" --n-main "$NMAIN"
}
main "$@"
exit $?
