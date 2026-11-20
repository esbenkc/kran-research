#!/bin/bash
# ONE laptop-queue job (labq) for the replication, 3 parallel lanes after the gates (Felix's final plan).
#   1. cache.py: this machine's streamed block-34 cache of Qwen3-8B (bf16, one block resident).
#   2. cache_check.py: G2(a) vs NDIF's dev rows (rel. err < 1e-2); abort on a fail. Also logs the sha256 against the
#      parent's mini cache (ref/parent/cache_sha256.json), report only.
#   3. gates.py: G1, calibrated G2(b), b on the train split, step-0 scan -> data/gates.json, data/scan.json.
#      A failed gate stops the job (frozen rule: no RL update runs).
#   4. lanes, each a background process (one GPU, shared; PREREG section 4, Felix plan v1 E6):
#        A = seeds 5,6,7 x sizes k0..k5 (main arm), then C2 (cue 0, 1,024 updates, seed 5), then S9 seed 5
#        B = C1 shuffled coin, seed 5, full 8,192, then S9 seed 6
#        C = C1 shuffled coin, seed 6, full 8,192, then S9 seed 7
#      S9 (exploratory, outside the status): seed s with its own fresh direction data/cue_u_{1120+s}.npy, k0..k5.
# Resumable: a finished run (data/runs*/<name>.json) is skipped; checkpoints (raw/ckpt) are copied to data/ckpt_resume/
# at the end, and restored at the start, so a re-queued job continues where this one stopped.
# Deadlines (unix seconds, env): LAST_START = no new run starts after it; STOP_AT = a running run checkpoints at its
# next frozen eval past it and exits 3 (resumable). Both default to "none".
# Usage: JOB=R1 LAST_START=<unix> STOP_AT=<unix> ./laptop_job.sh
#        SMOKE=1: Qwen3-0.6B, 16 updates per run, no gates, outputs in smoke/ (lane timing check).
#        VARIANT (DEVIATIONS P1, E7b, Tove #27078): full (default, plan v1 E6 above) |
#          e7b_mini   = ONE sequential lane on the mini (MACHINE=mini): cache, gates, mains 5-7 x k0..5, C2 s5, C1 s5
#          e7b_laptop = labq job: C1 s6 (lane B), then S9 seeds 5, 6, 7 (lane C); needs the mini's pushed data/gates.json
set -u
cd "$(dirname "$0")"
JOB=${JOB:-R1}
MACHINE=${MACHINE:-laptop}
VARIANT=${VARIANT:-full}
PFX=$([ "$MACHINE" = mini ] && echo m8 || echo l8)  # run-name prefix: m8 = mini, l8 = laptop (parent convention)
PY=${PY:-.venv/bin/python}  # the venv python directly: no 'uv run' per process (parent l8_c1_s1 died on a uv temp race)
LAST_START=${LAST_START:-99999999999}
STOP_AT=${STOP_AT:-0}
if [ -n "${SMOKE:-}" ]; then L=smoke/${MACHINE}_logs/$JOB; OUT=../smoke/runs; M8=Qwen/Qwen3-0.6B; EXTRA="--steps ${SMOKE_STEPS:-16}"
else L=data/$MACHINE/$JOB; OUT=runs; M8=Qwen/Qwen3-8B; EXTRA=""; fi
mkdir -p "$L" raw
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
S0=0.05469028438159526  # 33.299/608.865, PREREG grid s_0
share() { python3 -c "print(repr($S0 / 2**$1))"; }
TAG=$(basename "$M8" | tr 'A-Z' 'a-z')

log "START job $JOB variant $VARIANT machine $MACHINE commit $(git rev-parse --short HEAD 2>/dev/null) model $M8 last_start $LAST_START stop_at $STOP_AT"
if [ "$VARIANT" = e7b_laptop ] && [ -z "${SMOKE:-}" ] && [ ! -f data/gates.json ]; then
  log "ABORT: e7b_laptop needs the mini's data/gates.json in the checked-out commit (never re-solve the gates here)"; exit 1
fi
if [ -d data/ckpt_resume ] && [ -z "${SMOKE:-}" ]; then mkdir -p raw/ckpt && cp -R data/ckpt_resume/. raw/ckpt/ && log "restored checkpoints from data/ckpt_resume"; fi
if [ ! -f "raw/cache_$TAG/eager_rec_tail.npy" ]; then
  CSL_MODEL=$M8 $PY cache.py > "$L/cache_$TAG.log" 2>&1 || { log "FAILED cache.py $M8"; tail -20 "$L/cache_$TAG.log"; exit 1; }
  log "OK cache.py $M8"
fi
X8="--x-offset 0"
if [ -z "${SMOKE:-}" ]; then
  $PY cache_check.py "$L/cache_check.json" > "$L/cache_check.log" 2>&1
  [ $? -ne 0 ] && { log "ABORT: G2(a) failed on this machine's cache"; cat "$L/cache_check.log"; exit 1; }
  log "OK G2(a); sha vs parent mini cache: $(tail -1 "$L/cache_check.log")"
  if [ ! -f data/gates.json ]; then
    CSL_MODEL=$M8 $PY gates.py > "$L/gates.log" 2>&1 || { log "FAILED gates.py"; tail -20 "$L/gates.log"; exit 1; }
  fi
  ok=$($PY -c "import json; g=json.load(open('data/gates.json')); print(int(g['G1']['pass'] and g['G2']['pass']))")
  [ "$ok" = "1" ] || { log "ABORT: a gate failed (data/gates.json); no RL update runs"; grep -E '^G[12]' "$L/gates.log"; exit 0; }
  log "OK gates G1, G2 pass (data/gates.json)"
  X8="--x-offset gates"
fi

# entries: name|rl.py args
LANE_A=(); LANE_B=(); LANE_C=()
MAINS=(); for s in 5 6 7; do for k in 0 1 2 3 4 5; do MAINS+=("${PFX}_k${k}_s${s}|--share $(share $k) --k $k --seed $s $X8"); done; done
C2="${PFX}_c2_s5|--cue-alpha 0 --seed 5 --steps 1024 --full $X8"
C1_5="${PFX}_c1_s5|--arm shuf --share $(share 0) --k 0 --seed 5 --full $X8"
C1_6="${PFX}_c1_s6|--arm shuf --share $(share 0) --k 0 --seed 6 --full $X8"
s9() { for k in 0 1 2 3 4 5; do echo "${PFX}u$((1120 + $1))_k${k}_s$1|--share $(share $k) --k $k --seed $1 --u-file data/cue_u_$((1120 + $1)).npy --out-sub runs_explore $X8"; done; }
case "$VARIANT" in
  full)
    LANE_A=("${MAINS[@]}" "$C2"); while IFS= read -r e; do LANE_A+=("$e"); done < <(s9 5)
    LANE_B=("$C1_5"); while IFS= read -r e; do LANE_B+=("$e"); done < <(s9 6)
    LANE_C=("$C1_6"); while IFS= read -r e; do LANE_C+=("$e"); done < <(s9 7) ;;
  e7b_mini)
    LANE_A=("${MAINS[@]}" "$C2" "$C1_5") ;;
  e7b_laptop)
    LANE_B=("$C1_6")
    for s in 5 6 7; do while IFS= read -r e; do LANE_C+=("$e"); done < <(s9 $s); done ;;
  *) log "ABORT: unknown VARIANT $VARIANT"; exit 1 ;;
esac
if [ -n "${SMOKE:-}" ]; then  # smoke: first SMOKE_N entries of every lane; LANES picks which lanes run (timing check)
  [ ${#LANE_A[@]} -gt 0 ] && LANE_A=("${LANE_A[@]:0:${SMOKE_N:-2}}")
  [ ${#LANE_B[@]} -gt 0 ] && LANE_B=("${LANE_B[@]:0:${SMOKE_N:-2}}")
  [ ${#LANE_C[@]} -gt 0 ] && LANE_C=("${LANE_C[@]:0:${SMOKE_N:-2}}")
fi
case "$VARIANT" in e7b_mini) DEF_LANES=A ;; e7b_laptop) DEF_LANES=BC ;; *) DEF_LANES=ABC ;; esac
LANES=${LANES:-$DEF_LANES}

run_one() {  # $1 = lane, $2 = entry
  local lane=$1 name=${2%%|*} args=${2#*|} sub=$OUT rc t0
  case "$args" in *"--out-sub runs_explore"*) sub=$([ -n "${SMOKE:-}" ] && echo "../smoke/runs_explore" || echo runs_explore); args=${args/--out-sub runs_explore/};; esac
  [ -f "data/$sub/$name.json" ] && { log "[$lane] SKIP $name (done)"; return 0; }
  [ "$(date +%s)" -gt "$LAST_START" ] && { log "[$lane] LAST START passed: not starting $name"; return 4; }
  for attempt in 1 2; do
    log "[$lane] START $name (try $attempt) :: $args"
    t0=$(date +%s)
    CSL_MODEL=$M8 $PY rl.py --name "$name" $args --no-cast --head-dtype fp32 --machine "$MACHINE" --out-sub "$sub" \
       --stop-at "$STOP_AT" $EXTRA >> "$L/$name.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { log "[$lane] OK $name $(grep '^DONE' "$L/$name.log" | tail -1)"; return 0; }
    [ $rc -eq 3 ] && { log "[$lane] STOPPED $name at STOP_AT (resumable)"; return 3; }
    if [ $(( $(date +%s) - t0 )) -lt 30 ] && [ $attempt -eq 1 ]; then log "[$lane] $name exit $rc within 30 s: retry once"; sleep 5; continue; fi
    log "[$lane] FAILED $name (exit $rc)"; tail -20 "$L/$name.log"; return 1
  done
}
run_lane() {  # $1 = lane name, rest = entries
  local lane=$1; shift
  for e in "$@"; do
    run_one "$lane" "$e"; rc=$?
    [ $rc -eq 3 ] || [ $rc -eq 4 ] && break
  done
  log "[$lane] lane end"
}
PIDS=()
for ln in A B C; do
  case "$LANES" in *$ln*) ;; *) continue;; esac
  [ ${#PIDS[@]} -gt 0 ] && sleep 20  # stagger the model loads
  eval "run_lane $ln \"\${LANE_$ln[@]}\"" &
  PIDS+=($!)
done
wait "${PIDS[@]}"
if [ -z "${SMOKE:-}" ] && [ -d "raw/ckpt/$TAG" ]; then
  mkdir -p data/ckpt_resume && cp -R raw/ckpt/. data/ckpt_resume/ && log "saved checkpoints to data/ckpt_resume ($(du -sh data/ckpt_resume | cut -f1))"
fi
log "END job $JOB"
