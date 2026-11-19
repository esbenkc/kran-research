#!/bin/bash
# One laptop-queue job (or the Qwen3-0.6B smoke on the mini). Idempotent: every step is skipped when its output exists,
# so a restart resumes.
#   clean   extract clean per-token acts (all splits) -> fit.py fit (monitors, random direction, forecast.json)
#           -> score clean -> fit.py gate (PREREG clean validity gate; exit 2 = VOID, no fine-tune runs)
#   main S  attack.py seed S (300 steps, adapters at 150 and 300) -> extract step 150 (probe,refit,eval shards)
#           -> gate.py gate150 -> extract step 300 (shards only if step 150 failed the premise gate)
#           -> gate.py final (data/gate/main__seedS.json; prunes the unscored checkpoint's shards)
#   random S  attack.py --control random seed S -> extract the partner's scored step with refit,eval shards, the other
#           step scores-only (both with shards when the partner gate is not there yet) -> gate.py final
#   ctrlkl S  extract the panel-fine-tune-8b KL-only adapter (data/ctrlkl/, step 300) with eval shards
# Usage: ./run_all.sh clean 'main 0' 'main 1' 'main 2'                 (job J1)
#        ./run_all.sh clean 'random 0' 'random 1' 'ctrlkl 0' 'ctrlkl 1' (job J2; 'clean' fetches J1's pushed fits)
# Env: MP_MODEL / MP_DATA / GATE_SKIP=1 / STEPS (smoke only), JOB=<tag> (logs in data/logs/<tag>/).
set -u
cd "$(dirname "$0")"
D=${MP_DATA:-data}
L="$D/logs${JOB:+/$JOB}"
mkdir -p "$L"
PY=${PY:-"uv run --frozen python"}
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/chain.log"; }
step() {  # step <logname> <args...>
  local n=$1; shift
  log "START $n"
  $PY "$@" > "$L/$n.log" 2>&1 || { log "FAILED $n (tail below)"; tail -20 "$L/$n.log"; exit 1; }
  log "OK $n"
}
BRANCH=${BRANCH:-swarm/oracle-1119-max-pool-8b}
for spec in "$@"; do
  set -- $spec; what=$1; seed=${2:-}
  case $what in
  clean)
    # labq checks out the SUBMITTED commit: a later job takes the clean state + fits that an earlier job pushed
    if [ ! -f "$D/fits/probes.npz" ] && [ -z "${MP_DATA:-}" ]; then
      git fetch -q origin "$BRANCH" 2>/dev/null \
        && git checkout FETCH_HEAD -- data/fits data/states/clean data/results/forecast.json 2>/dev/null \
        && log "fetched clean state + fits from origin ($(git rev-parse --short FETCH_HEAD))" \
        || log "no pushed clean fits: fitting here"
    fi
    [ -f "$D/states/clean/lp.npz" ] || step extract_clean extract.py --state clean
    [ -f "$D/results/forecast.json" ] || step fit fit.py fit
    [ -f "$D/states/clean/scores.npz" ] || step score_clean extract.py --state clean --score-only
    $PY fit.py gate > "$L/gate_clean.log" 2>&1; g=$?; log "$(tail -1 "$L/gate_clean.log")"
    if [ $g -eq 2 ]; then
      [ "${GATE_SKIP:-0}" = 1 ] && log "GATE_SKIP=1 (smoke only): continuing past VOID" || { log "VOID: stop before any fine-tune"; exit 2; }
    elif [ $g -ne 0 ]; then log "FAILED clean gate"; tail -20 "$L/gate_clean.log"; exit 1; fi
    ;;
  main|random)
    name="${what}__seed$seed"
    ctl=none; [ "$what" = random ] && ctl=random
    keep=probe,refit,eval; [ "$what" = random ] && keep=refit,eval
    [ -f "$D/runs/$name.json" ] || step "attack_$name" attack.py --seed "$seed" --control "$ctl" ${STEPS:+--steps $STEPS}
    if [ -f "$D/gate/$name.json" ]; then log "SKIP $name (gate done)"; continue; fi
    k150=$keep; k300=$keep
    if [ "$what" = main ]; then
      [ -f "$D/states/${name}__step150/meta.json" ] || step "extract_${name}__step150" extract.py --state "${name}__step150" \
        --adapter "$D/runs/${name}__step150__adapter" --keep $k150
      g=$($PY gate.py gate150 main "$seed" 2>/dev/null | tail -1); log "premise gate step 150 $name: $g"
      [ "$g" = pass ] && k300=""
    else
      p=$($PY gate.py partner "$seed" 2>/dev/null | tail -1); log "partner pick for $name: $p"
      [ "$p" = 150 ] && k300=""
      [ "$p" = 300 ] && k150=""
      [ -f "$D/states/${name}__step150/meta.json" ] || step "extract_${name}__step150" extract.py --state "${name}__step150" \
        --adapter "$D/runs/${name}__step150__adapter" --keep "$k150"
    fi
    [ -f "$D/states/${name}__step300/meta.json" ] || step "extract_${name}__step300" extract.py --state "${name}__step300" \
      --adapter "$D/runs/${name}__step300__adapter" --keep "$k300"
    step "gate_$name" gate.py final "$what" "$seed"
    log "DONE $name"
    ;;
  ctrlkl)
    name="ctrlkl__lam10000__seed$seed"
    [ -f "$D/states/$name/meta.json" ] || step "extract_$name" extract.py --state "$name" \
      --adapter "data/ctrlkl/${name}__adapter" --keep eval
    log "DONE $name"
    ;;
  *) log "unknown spec $spec"; exit 1 ;;
  esac
done
log "CHAIN_DONE"
