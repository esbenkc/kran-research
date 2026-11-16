#!/bin/bash
# One laptop-queue job (labq). Idempotent and resumable: finished runs (data/runs/<name>.json) are skipped, a run cut by
# the job deadline resumes from raw/ckpt/ if the clone survives (rl.py --ckpt-every 512 / --stop-at).
#   1. cache.py: this machine's streamed block-34 cache (bf16, one block resident; same code as the mini).
#   2. cache_check.py: G2(a) against NDIF's dev rows (rel. err < 1e-2), abort on a fail (PREREG G2, laptop clause);
#      sha256 vs the mini's cache: same -> b from data/gates.json (--x-offset gates); differs -> b solved on this cache's
#      train split (--x-offset solve), logged in the run JSON.
#   3. the run list for this JOB, serially, --machine laptop, run names l8_* (8B -> data/runs/) and l06_* (0.6B matched
#      arm -> data/runs_06b/), Felix #22636 contract. J1 = 8B seeds 3-4 + C1 s1, s0 + C2; J2 = 8B seeds 0-2 (laptop copy:
#      replication + the PREREG laptop-join check). 0.6B seeds 3-4 run on the mini (Qwen3-0.6B is not in the laptop's
#      HF cache; no download on its disk).
# Self-stop: no run starts after START+7.5 h; a running run checkpoints and exits at START+7.8 h.
# Usage: JOB=J1 ./laptop_job.sh     JOB=J2 ./laptop_job.sh     (SMOKE=1: Qwen3-0.6B, 16 updates per run, smoke/laptop/)
set -u
cd "$(dirname "$0")"
JOB=${JOB:?set JOB=J1 or J2}
PY=${PY:-"uv run --frozen python"}
L=data/laptop/$JOB; [ -n "${SMOKE:-}" ] && L=smoke/laptop_logs/$JOB
mkdir -p "$L" raw
START=$(date +%s); LAST_START=$((START + 27000)); STOP_AT=$((START + 28080))
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
S0=0.05469028438159526  # 33.299/608.865, PREREG grid s_0
share() { python3 -c "print(repr($S0 / 2**$1))"; }

if [ -n "${SMOKE:-}" ]; then OUT=../smoke/laptop; EXTRA="--steps 16"; else OUT=runs; EXTRA=""; fi

log "START job $JOB commit $(git rev-parse --short HEAD 2>/dev/null)"
build_cache() {  # $1 = HF id
  local tag; tag=$(basename "$1" | tr 'A-Z' 'a-z')
  [ -f "raw/cache_$tag/eager_rec_tail.npy" ] && return 0
  CSL_MODEL=$1 $PY cache.py > "$L/cache_$tag.log" 2>&1 || { log "FAILED cache.py $1"; tail -20 "$L/cache_$tag.log"; exit 1; }
  log "OK cache.py $1"
}
X8="--x-offset 0"
if [ -z "${SMOKE:-}" ]; then
  build_cache Qwen/Qwen3-8B
  $PY cache_check.py "$L/cache_check.json" > "$L/cache_check.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { log "ABORT: G2(a) failed on this machine's cache (exit $rc)"; cat "$L/cache_check.log"; exit 1; }
  if [ "$(tail -1 "$L/cache_check.log")" = "same" ]; then X8="--x-offset gates"; else X8="--x-offset solve"; fi
  log "OK cache_check: $(tail -1 "$L/cache_check.log") as the mini -> 8B runs use $X8"
fi
M8=Qwen/Qwen3-8B; [ -n "${SMOKE:-}" ] && M8=Qwen/Qwen3-0.6B  # smoke: every run on 0.6B

# entries: name|HF model|rl.py args (8B: + $X8; 0.6B matched arm: b = 0)
RUNS=()
case $JOB in
  J1) for s in 3 4; do for k in 0 1 2 3 4 5; do RUNS+=("l8_k${k}_s${s}|$M8|--share $(share $k) --k $k --seed $s $X8"); done; done
      RUNS+=("l8_c1_s1|$M8|--arm shuf --share $(share 0) --k 0 --seed 1 --full $X8")
      RUNS+=("l8_c1_s0|$M8|--arm shuf --share $(share 0) --k 0 --seed 0 --full $X8")
      RUNS+=("l8_c2_s0|$M8|--cue-alpha 0 --seed 0 --steps 1024 --full $X8") ;;
  J2) for s in 0 1 2; do for k in 0 1 2 3 4 5; do RUNS+=("l8_k${k}_s${s}|$M8|--share $(share $k) --k $k --seed $s $X8"); done; done
 ;;
  *) log "unknown JOB $JOB"; exit 1 ;;
esac

for r in "${RUNS[@]}"; do
  name=${r%%|*}; rest=${r#*|}; model=${rest%%|*}; args=${rest#*|}
  sub=$OUT; case "$args" in *"--out-sub runs_06b"*) [ -z "${SMOKE:-}" ] && sub=runs_06b; args=${args/--out-sub runs_06b/};; esac
  [ -f "data/$sub/$name.json" ] && { log "SKIP $name (done)"; continue; }
  [ "$(date +%s)" -gt "$LAST_START" ] && { log "LAST START passed: not starting $name"; break; }
  build_cache "$model"
  log "START $name [$model] :: $args"
  CSL_MODEL=$model $PY rl.py --name "$name" $args --no-cast --head-dtype fp32 --machine laptop --out-sub "$sub" \
     --stop-at "$STOP_AT" $EXTRA > "$L/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 3 ]; then log "STOPPED $name at the job deadline (resumable)"; break; fi
  [ $rc -ne 0 ] && { log "FAILED $name (exit $rc)"; tail -20 "$L/$name.log"; continue; }
  log "OK $name $(grep '^DONE' "$L/$name.log" | tail -1)"
done
log "END job $JOB"
