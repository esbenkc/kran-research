#!/bin/bash
# theory-1122 continuation, DEVIATIONS P4: two extra cells on the Mac mini, ONE solo heavy job at a time (Vera gate #28229).
#   J2 first (Tove #28225): m8_c2_s6  = C2 pipeline check, cue 0, 1,024 updates, seed 6 (outside the status; c2_check only)
#   J1 then:                m8u1127_k5_s7 = S9 fresh direction u1127, k5, seed 7 (exploratory, runs_explore only)
# Same rl.py arguments as laptop_job.sh builds for these cells (MACHINE=mini). Wall stop 09:15 PST (resumable).
# Never edit this file while it runs (bash reads by byte offset).
main() {
  set -u
  cd "$(dirname "$0")"
  local L=data/mini/M3 PY=.venv/bin/python
  local WALL; WALL=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-22 09:15:00" +%s)
  mkdir -p "$L"
  log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
  log "START M3 (DEVIATIONS P4) commit $(git rev-parse --short HEAD) cache sha256 $($PY -c "import numpy as np,hashlib;print(hashlib.sha256(np.load('raw/cache_qwen3-8b/eager_rec_tail.npy').tobytes()).hexdigest())")"
  log "memory: $(memory_pressure | tail -1)"
  local S5; S5=$(python3 -c "print(repr(0.05469028438159526 / 2**5))")
  run() {  # $1 name, $2 out-sub, rest = args
    local name=$1 sub=$2; shift 2
    [ -f "data/$sub/$name.json" ] && { log "SKIP $name (done)"; return 0; }
    log "START $name :: $*"
    CSL_MODEL=Qwen/Qwen3-8B /usr/bin/time -l $PY rl.py --name "$name" "$@" --no-cast --head-dtype fp32 --machine mini \
      --out-sub "$sub" --wall-stop "$WALL" >> "$L/$name.log" 2>&1
    local rc=$?
    log "END $name exit $rc $(grep '^DONE' "$L/$name.log" | tail -1) peak $(grep 'peak memory' "$L/$name.log" | tail -1)"
    return $rc
  }
  run m8_c2_s6 runs --cue-alpha 0 --seed 6 --steps 1024 --full --x-offset gates
  run m8u1127_k5_s7 runs_explore --share "$S5" --k 5 --seed 7 --u-file data/cue_u_1127.npy --x-offset gates
  log "END M3"
}
main "$@"
exit
