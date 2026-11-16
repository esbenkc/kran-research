#!/bin/bash
# One laptop-queue job (or the Qwen3-0.6B smoke on the mini): idempotent, resumes after a restart.
#   1. clean state: extract -> panel fit (C rule) -> score -> z-moments -> score   (skipped when data/fits exist)
#   2. per run spec "<control> <seed>" (control = none | kl | random; lambda 1e4 always):
#      attack_panel.py -> extract (step 150 and 300) -> panel.py score -> refit.py (step 300 only) -> drop acts
# Usage: ./run_all.sh 'none 0' 'none 1' 'none 2'        (job A)
#        ./run_all.sh 'random 0' 'random 1' 'kl 0' 'kl 1' (job B)
# Env: PFT_MODEL / PFT_DATA / GATE_SKIP=1 (smoke only), STEPS (smoke only; default 300), KEEP_ACTS=1 keeps attacked
#      acts, BESTSINGLE=1 runs the PREREG s8 best-single-readout fits on the clean state at the end.
set -u
cd "$(dirname "$0")"
D=${PFT_DATA:-data}
# JOB=<tag>: logs in data/logs/<tag>/ and checksums in data/acts_sha256_<tag>.txt, so two laptop jobs never add the
# same file with different content (labq rebases each job's output commit onto the branch; Kasper #22374).
L="$D/logs${JOB:+/$JOB}"
SHA="$D/acts_sha256${JOB:+_$JOB}.txt"
mkdir -p "$L"
PY=${PY:-"uv run --frozen python"}
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
step() {  # step <logname> <args...>
  local n=$1; shift
  log "START $n"
  $PY "$@" > "$L/$n.log" 2>&1 || { log "FAILED $n (tail below)"; tail -20 "$L/$n.log"; exit 1; }
  log "OK $n"
}
# labq checks out the SUBMITTED commit: a later job fetches the clean fits that an earlier job pushed (public repo).
if [ ! -f "$D/fits/zmoments.json" ] && [ -z "${PFT_DATA:-}" ]; then
  git fetch -q origin swarm/oracle-1116-panel-fine-tune-8b 2>/dev/null \
    && git checkout FETCH_HEAD -- data/fits data/scores/clean.npz data/states/clean 2>/dev/null \
    && log "fetched clean fits from origin ($(git rev-parse --short FETCH_HEAD))" || log "no pushed clean fits: fitting here"
fi
if [ ! -f "$D/fits/zmoments.json" ]; then
  [ -f "$D/acts/clean/cty.npy" ] || step extract_clean extract.py --state clean --all
  [ -f "$D/fits/panel.npz" ] || step fit_panel panel.py fit
  step score_clean0 panel.py score clean
  step zmoments panel.py zmoments
fi
[ -f "$D/scores/clean.npz" ] && $PY -c "import numpy as np,sys; z=np.load('$D/scores/clean.npz'); sys.exit(0 if 'panelmax' in z and 'panelmax_rescaled' in z else 1)" \
  || step score_clean panel.py score clean
# PREREG s6 GATE: clean max-panel >= 90% at its own line, else VOID (no fine-tune runs)
$PY panel.py gate > "$L/gate.log" 2>&1; g=$?; log "$(tail -1 "$L/gate.log")"
[ $g -eq 2 ] && [ "${GATE_SKIP:-0}" = 1 ] && log "GATE_SKIP=1 (smoke only): continuing past VOID"
[ $g -eq 2 ] && [ "${GATE_SKIP:-0}" != 1 ] && { log "VOID: stop before any fine-tune"; exit 2; }
[ $g -eq 2 ] && g=0
[ $g -ne 0 ] && { log "FAILED gate"; tail -20 "$L/gate.log"; exit 1; }
for spec in "$@"; do
  set -- $spec; ctl=$1; seed=$2
  pre=panel; [ "$ctl" != none ] && pre=ctrl$ctl
  name="${pre}__lam10000__seed$seed"
  [ -f "$D/runs/$name.json" ] || step "attack_$name" attack_panel.py --lam 1e4 --seed "$seed" --control "$ctl" ${STEPS:+--steps $STEPS}
  for st in "$name" "${name}__step150"; do
    ad="$D/runs/${name}__adapter"; [ "$st" != "$name" ] && ad="$D/scratch/${name}__step150__adapter"
    [ -d "$ad" ] || { log "SKIP $st (no adapter)"; continue; }
    [ -f "$D/states/$st/meta.json" ] && [ -f "$D/scores/$st.npz" ] && { [ "$st" != "$name" ] || [ -f "$D/refit/$st.json" ]; } && continue
    [ -f "$D/acts/$st/cty.npy" ] || step "extract_$st" extract.py --state "$st" --adapter "$ad"
    step "score_$st" panel.py score "$st"
    [ "$st" = "$name" ] && step "refit_$st" refit.py "$st"
    if [ "${KEEP_ACTS:-0}" != 1 ]; then (cd "$D" && shasum -a 256 "acts/$st/cty.npy") >> "$SHA"; rm -rf "$D/acts/$st"; fi
  done
  log "DONE $name"
done
if [ "${BESTSINGLE:-0}" = 1 ] && [ ! -f "$D/fits/bestsingle.json" ] && [ -f "$D/acts/clean/cty.npy" ]; then
  step bestsingle panel.py bestsingle
fi
log "CHAIN_DONE"
