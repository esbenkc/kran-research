#!/bin/bash
# One laptop-queue job (labq). Idempotent: every finished stage/set is skipped, so a rerun resumes.
#   SEEDS="0" ./run.sh          job A: base model + strip seed 0 + eval seed 0
#   FETCH=1 SEEDS="1 2" ./run.sh   job B: first pulls the branch head (job A's data/base), then seeds 1 and 2
# Smoke on the mini: GRA_MODEL=Qwen/Qwen3-0.6B GRA_OUT=smoke SEEDS=0 MAXSTEPS=3 ./run.sh
set -u
cd "$(dirname "$0")"
export GRA_MODEL=${GRA_MODEL:-Qwen/Qwen3-8B} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=0
OUTD=${GRA_OUT:-data}; export GRA_OUT=$OUTD
SEEDS=${SEEDS:-0}
BR=swarm/oracle-1118-gated-refusal
PY=${PY:-"uv run --frozen --no-sync python"}
JOB=${LABQ_JOB_ID:-local_$(date +%m%d_%H%M%S)}
L=$OUTD/logs/$JOB; mkdir -p "$L"
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
if [ "${FETCH:-0}" = 1 ]; then
  git fetch -q origin "$BR" && git reset -q --hard FETCH_HEAD && log "fetched $BR at $(git rev-parse --short HEAD)"
  mkdir -p "$L"
fi
log "START $JOB model $GRA_MODEL seeds '$SEEDS' commit $(git rev-parse --short HEAD 2>/dev/null) out $OUTD"
/usr/bin/time -l $PY job.py base > "$L/base.log" 2>&1 || { log "FAILED base"; tail -30 "$L/base.log"; exit 1; }
log "base: $(grep -E 'base done|str refusal' "$L/base.log" | tr '\n' ' ')"
for s in $SEEDS; do
  if [ ! -f "$OUTD/strip_s$s/train_log.json" ] || [ ! -d "adapters/strip_s$s" ]; then
    /usr/bin/time -l $PY strip.py --seed "$s" ${MAXSTEPS:+--max_steps $MAXSTEPS} > "$L/strip_s$s.log" 2>&1 \
      || { log "FAILED strip s$s"; tail -30 "$L/strip_s$s.log"; exit 1; }
    log "strip s$s: $(tail -1 "$L/strip_s$s.log"); peak $(grep 'maximum resident' "$L/strip_s$s.log")"
  fi
  /usr/bin/time -l $PY job.py eval --seed "$s" > "$L/eval_s$s.log" 2>&1 || { log "FAILED eval s$s"; tail -30 "$L/eval_s$s.log"; exit 1; }
  log "eval s$s: $(grep -E 'gate:|push pick|regen|str refusal' "$L/eval_s$s.log" | tr '\n' ' ')"
done
log "DONE $JOB"
