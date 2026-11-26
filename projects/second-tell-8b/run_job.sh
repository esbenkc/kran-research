#!/bin/bash
# The whole second-tell-8b run on the Mac mini, ONE process at a time (PREREG section 4; Vera's compute gate):
#   1. stage 1: MAIN, SHUF-A, COIN x seeds 5-9 (15 tails, one batched job), cue A at k1, L = 256
#   2. stage 2 k4: the 15 stage-1 tails + FRESH 5-9 (20 tails), cue B at k4, cap 1,536  -> the status
#   3. stage 2 k2: same tails from the same stage-1 adapters, cue B at k2                 -> law read only (S1)
# Each step resumes from raw/ckpt/ if a checkpoint exists; a finished step (its adapters_sha256.json) is skipped.
# Logs: data/mini/<step>.log. Usage: bash run_job.sh
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
L=data/mini
mkdir -p "$L"
log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$L/chain.log"; }
S1=""; S2=""
for s in 5 6 7 8 9; do S1="$S1 main:$s shuf:$s coin:$s"; S2="$S2 main:$s shuf:$s coin:$s fresh:$s"; done
log "START commit $(git rev-parse --short HEAD) host $(hostname -s)"
run() {  # name, done-marker, args...
  local name=$1 marker=$2; shift 2
  if [ -f "$marker" ]; then log "SKIP $name (done: $marker)"; return 0; fi
  log "RUN $name: tt.py $*"
  /usr/bin/time -l $PY tt.py "$@" > "$L/$name.log" 2>&1
  local rc=$?
  log "END $name rc $rc peak $(grep 'peak memory footprint' "$L/$name.log" | awk '{print $1}')"
  return $rc
}
run stage1 data/stage1/adapters_sha256.json --stage 1 --runs $S1 --n1 256 --kA 1 --eval-bs 8 || exit 1
run k4 data/k4/adapters_sha256.json --stage 2 --kB 4 --runs $S2 --n2 1536 --kA 1 --eval-bs 8 || exit 1
run k2 data/k2/adapters_sha256.json --stage 2 --kB 2 --runs $S2 --n2 1536 --kA 1 --eval-bs 8 || exit 1
log "ALL DONE"
