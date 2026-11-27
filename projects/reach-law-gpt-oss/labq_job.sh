#!/bin/bash
# One laptop-queue job (room defense-1127-reach-law-gpt-oss). Runs in projects/reach-law-gpt-oss of labq's clean clone.
#   bash labq_job.sh <filler-stop-at ISO>
# Checks the harness tests, then runs every cell in PREREG order (gate, main, prompt, filler). Only the filler stops at
# the given time, and only when the job started before it (DEVIATIONS v1.2).
# Resumable: a run whose raw file exists is skipped (a resubmitted job at a later commit skips every pushed run).
# Outputs: data/gpt-oss-20b/ only. No API keys, no network after the model is cached.
main() {
  local STOP=$1
  uv sync --frozen || return 6
  uv run python test_arms.py > data/gpt-oss-20b-test_arms.txt 2>&1 || { cat data/gpt-oss-20b-test_arms.txt; return 7; }
  caffeinate -i uv run python run.py --model gpt-oss-20b --phase all --filler-stop-at "$STOP"
}
mkdir -p data
main "$@"
exit $?
