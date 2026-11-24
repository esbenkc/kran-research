#!/bin/bash
# J1 (Bruno #29950, Vera compute gate #29955): the RL chain on the Mac mini, ONE heavy job, sequential.
# Parent rl.py setting (fp32 tail + fp32 head, --no-cast, --x-offset gates = parent G1.b) + LoRA saved every 8 updates.
# Order: k2 mains (s5-7), C1 (s5-7), k3 mains, k4 mains. Steps: >= the size's last grid point + 8 (PREREG (2)).
# Never edit this file while it runs (bash reads by byte offset).
main() {
  set -u
  cd "$(dirname "$0")"
  local L=data/mini/J1 PY=.venv/bin/python
  mkdir -p "$L"
  log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
  log "START J1 commit $(git rev-parse --short HEAD) cache sha256 $($PY -c "import numpy as np,hashlib;print(hashlib.sha256(np.load('raw/cache_qwen3-8b/eager_rec_tail.npy').tobytes()).hexdigest())")"
  log "memory: $(memory_pressure | tail -1)"
  local S0=0.05469028438159526
  share() { python3 -c "print(repr($S0 / 2**$1))"; }
  run() {  # $1 name, rest = args
    local name=$1; shift
    [ -f "data/runs/$name.json" ] && { log "SKIP $name (done)"; return 0; }
    log "START $name :: $*"
    CSL_MODEL=Qwen/Qwen3-8B /usr/bin/time -l $PY rl.py --name "$name" "$@" --no-cast --head-dtype fp32 --machine mini \
      --x-offset gates --ckpt-every 0 --out-sub runs >> "$L/$name.log" 2>&1
    local rc=$?
    log "END $name exit $rc $(grep '^DONE' "$L/$name.log" | tail -1) | $(grep 'peak memory footprint' "$L/$name.log" | tail -1) | $(memory_pressure | tail -1)"
    return $rc
  }
  for s in 5 6 7; do run m8_k2_s$s --share "$(share 2)" --k 2 --seed $s --steps 96 --lora-until 88; done
  for s in 5 6 7; do run m8_c1_s$s --arm shuf --share "$(share 0)" --k 0 --seed $s --full --steps 384 --lora-until 352; done
  for s in 5 6 7; do run m8_k3_s$s --share "$(share 3)" --k 3 --seed $s --steps 168 --lora-until 160; done
  for s in 5 6 7; do run m8_k4_s$s --share "$(share 4)" --k 4 --seed $s --steps 360 --lora-until 352; done
  log "END J1"
}
main "$@"
exit
