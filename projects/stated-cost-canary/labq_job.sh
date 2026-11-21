#!/bin/bash
# One laptop-queue job (room defense-1121-stated-cost). Runs in projects/stated-cost-canary of labq's clean clone.
#   bash labq_job.sh <tag> <priorities> <max_hours>
#   Job A: bash labq_job.sh gpt-oss-20b-hidden P1 2.5      (N + C interleaved, 120 runs)
#   Job B: bash labq_job.sh gpt-oss-20b-visible P2,P3 1.0  (NV then CV, 120 runs)
# It first moves to the branch head (so a resumed job skips every run an earlier job pushed), refuses to run if the
# harness differs from the frozen code commit, then runs web/run.py under caffeinate -i. PY=echo = dry run.
# The whole body is one function, so bash has read all of it before `git reset` can change this file.
main() {
  local CODE_SHA=94368632f BR=swarm/defense-1121-stated-cost
  git fetch -q origin "$BR" && git reset -q --hard FETCH_HEAD || { echo "labq_job: fetch/reset failed"; return 5; }
  echo "labq_job: resume base $(git rev-parse HEAD)"
  if ! git diff --quiet "$CODE_SHA" HEAD -- web ':(exclude)web/out' pyproject.toml uv.lock; then
    echo "labq_job: harness drift vs $CODE_SHA"; git diff --stat "$CODE_SHA" HEAD -- web ':(exclude)web/out' pyproject.toml uv.lock
    return 4
  fi
  uv sync --frozen || return 6
  caffeinate -i ${PY:-uv run python} web/run.py --model gpt-oss-20b --tag "$1" --only "$2" --max-hours "$3"
}
main "$@"
exit $?
