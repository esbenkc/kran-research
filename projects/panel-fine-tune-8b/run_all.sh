#!/bin/bash
# One laptop-queue job (or the Qwen3-0.6B smoke on the mini): idempotent, resumes after a restart.
#   1. clean state: extract -> panel fit (C rule) -> score -> z-moments -> score   (skipped when data/fits exist)
#   2. per run spec "<control> <seed>" (control = none | kl | random; lambda 1e4 always):
#      attack_panel.py -> extract (step 150 and 300) -> panel.py score -> refit.py (step 300 only) -> drop acts
# Usage: ./run_all.sh 'none 0' 'none 1' 'none 2'        (job A)
#        ./run_all.sh 'random 0' 'random 1' 'kl 0' 'kl 1' (job B)
# Env: PFT_MODEL / PFT_DATA (smoke), STEPS (smoke only; default 300), KEEP_ACTS=1 keeps attacked acts.
set -u
cd "$(dirname "$0")"
D=${PFT_DATA:-data}
mkdir -p "$D/logs"
PY=${PY:-"uv run --frozen python"}
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$D/logs/chain.log"; }
step() {  # step <logname> <args...>
  local n=$1; shift
  log "START $n"
  $PY "$@" > "$D/logs/$n.log" 2>&1 || { log "FAILED $n (tail below)"; tail -20 "$D/logs/$n.log"; exit 1; }
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
[ -f "$D/scores/clean.npz" ] && $PY -c "import numpy as np,sys; sys.exit(0 if 'panelmax' in np.load('$D/scores/clean.npz') else 1)" \
  || step score_clean panel.py score clean
for spec in "$@"; do
  set -- $spec; ctl=$1; seed=$2
  pre=panel; [ "$ctl" != none ] && pre=ctrl$ctl
  name="${pre}__lam10000__seed$seed"
  [ -f "$D/runs/$name.json" ] || step "attack_$name" attack_panel.py --lam 1e4 --seed "$seed" --control "$ctl" ${STEPS:+--steps $STEPS}
  for st in "$name" "${name}__step150"; do
    ad="$D/runs/${name}__adapter"; [ "$st" != "$name" ] && ad="$D/runs/${name}__step150__adapter"
    [ -d "$ad" ] || { log "SKIP $st (no adapter)"; continue; }
    [ -f "$D/states/$st/meta.json" ] && [ -f "$D/scores/$st.npz" ] && { [ "$st" != "$name" ] || [ -f "$D/refit/$st.json" ]; } && continue
    [ -f "$D/acts/$st/cty.npy" ] || step "extract_$st" extract.py --state "$st" --adapter "$ad"
    step "score_$st" panel.py score "$st"
    [ "$st" = "$name" ] && step "refit_$st" refit.py "$st"
    if [ "${KEEP_ACTS:-0}" != 1 ]; then shasum -a 256 "$D/acts/$st/cty.npy" >> "$D/acts_sha256.txt"; rm -rf "$D/acts/$st"; fi
  done
  log "DONE $name"
done
if [ "${BESTSINGLE:-0}" = 1 ] && [ ! -f "$D/fits/bestsingle.json" ] && [ -f "$D/acts/clean/cty.npy" ]; then
  step bestsingle panel.py bestsingle
fi
log "CHAIN_DONE"
