#!/bin/bash
# One laptop-queue box (labq) of the whole 8B run. Idempotent and resumable: every finished cell (data/runs/<name>.json,
# data/audit/s<seed>.json, data/g0.json) is skipped; a run cut by the box deadline saved data/ckpt/<name>.pt and the next
# box resumes it bit for bit. labq --cmd: git fetch + reset to the branch head (gets every earlier box's outputs), then
#   uv sync --frozen --no-install-project && HOURS=1 caffeinate -i ./run.sh
# Order (PREREG FROZEN RULE): cache (+ sha check vs the first box) -> G0 -> switch s0, s1, s2 (+ s3, s4 as replacements)
#   -> audit + sets per formed seed -> retrain SET x3, ALL x3, TOP-1 x3, LOO x3, RANDOM draws 1-2 x first 2 formed seeds.
# Smoke on the mini: HC_MODEL=Qwen/Qwen3-0.6B HC_OUT=smoke/q06 EVAL=16 STOPAT=0.0 CAP=64 HOURS=0.2 ./run.sh
set -u
cd "$(dirname "$0")"
export HC_MODEL=${HC_MODEL:-Qwen/Qwen3-8B} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUTD=${HC_OUT:-data}; export HC_OUT=$OUTD
PY=${PY:-"uv run --frozen --no-sync python"}
HOURS=${HOURS:-1}; EVAL=${EVAL:-64}; STOPAT=${STOPAT:-0.75}; CAP=${CAP:-4096}
JOB=${LABQ_JOB_ID:-local_$(date +%m%d_%H%M%S)}
L=$OUTD/laptop/$JOB; mkdir -p "$L" raw
START=$(date +%s); DEADLINE=$(python3 -c "print(int($START + $HOURS * 3600 - 240))")
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
SHORT=$(basename "$HC_MODEL")
CD=raw/step2/$SHORT/compile_C
log "START box $JOB model $HC_MODEL commit $(git rev-parse --short HEAD 2>/dev/null) hours $HOURS out $OUTD"

# --- cache: the parent's method (stock torch.compile, read-only q_norm hook at block n_layers//2, 704 prompts) ---
cp inputs/prompts.jsonl raw/prompts.jsonl
if [ ! -f "$CD/probe_rec_tail.npy" ]; then
  LAYER=$($PY -c "import cfgm; print(cfgm.PROBE_LAYER)" 2>/dev/null | tail -1)
  log "cache build: probe hook block $LAYER"
  /usr/bin/time -l $PY step1_torch.py --model "$HC_MODEL" --layer "$LAYER" --backend compile --tag C --split all --n -1 \
    --one-rep --passes rec --builds none,probe --out raw/step2 > "$L/cache.log" 2>&1 \
    || { log "FAILED cache build"; tail -20 "$L/cache.log"; exit 1; }
  log "cache built: $(grep -E 'none rec|probe rec' "$L/cache.log" | tr '\n' ' ')"
fi
SHA=$(cd "$CD" && shasum -a 256 none_rec_tail.npy probe_rec_tail.npy)
if [ -f "$OUTD/cache_sha.txt" ]; then
  [ "$SHA" = "$(cat "$OUTD/cache_sha.txt")" ] || { log "ABORT: cache sha differs from the first box"; echo "$SHA" > "$L/cache_sha_mismatch.txt"; exit 1; }
  log "cache sha = first box"
else
  echo "$SHA" > "$OUTD/cache_sha.txt"; log "cache sha written (first box)"
fi

# --- G0 ---
if [ ! -f "$OUTD/g0.json" ]; then
  $PY g0.py > "$L/g0.log" 2>&1; rc=$?
  log "G0 rc $rc: $(tail -1 "$L/g0.log")"
  [ $rc -ne 0 ] && [ $rc -ne 2 ] && { log "FAILED g0.py"; tail -20 "$L/g0.log"; exit 1; }
fi
D_ANS=$(python3 -c "import json; print(json.load(open('$OUTD/g0.json'))['d_ans'])")
python3 -c "import sys; sys.exit(0 if $D_ANS >= 0.5 else 1)" || { log "G0 NULL: d34 = $D_ANS < 0.5, no trace at the answer position; stop"; exit 0; }
log "G0 d34 = $D_ANS"

ADPD=$([ "$OUTD" = data ] && echo adapters || echo "$OUTD/adapters")
trap '[ -d "$ADPD" ] && (cd "$ADPD" && shasum -a 256 *_adapter.pt > SHA256SUMS 2>/dev/null)' EXIT
# run <name> <train.py args...>: skip if done; stop the box when too little time is left or at the deadline
run() {
  local n=$1; shift
  [ -f "$OUTD/runs/$n.json" ] && return 0
  [ "$(left)" -lt 300 ] && { log "BOX END: not starting $n ($(left) s left)"; exit 0; }
  log "START $n :: $*"
  $PY train.py --name "$n" "$@" --eval-every "$EVAL" --stop-at "$STOPAT" --deadline "$DEADLINE" > "$L/$n.log" 2>&1
  local rc=$?
  [ $rc -eq 3 ] && { log "BOX END: $n saved a checkpoint at the deadline ($(grep DEADLINE "$L/$n.log"))"; exit 0; }
  [ $rc -ne 0 ] && { log "FAILED $n (rc $rc)"; tail -20 "$L/$n.log"; exit 1; }
  log "OK $(grep '^DONE' "$L/$n.log" | tail -1)"
}
field() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }

# --- switch models: s0, s1, s2; s3, s4 replace seeds that do not form, until 3 have formed ---
FORMED=()
for s in 0 1 2 3 4; do
  [ ${#FORMED[@]} -ge 3 ] && break
  run "switch_s$s" --mode switch --seed $s --steps "$CAP"
  if [ "$(field "$OUTD/runs/switch_s$s.json" "d['formed']")" = "True" ]; then FORMED+=($s); fi
  log "switch_s$s formed=$(field "$OUTD/runs/switch_s$s.json" "d['formed']") T_f=$(field "$OUTD/runs/switch_s$s.json" "d['t_f']") formed so far: ${FORMED[*]:-none}"
done
echo "${FORMED[*]:-}" > "$OUTD/formed_seeds.txt"
[ ${#FORMED[@]} -le 1 ] && { log "FORMATION NULL: ${#FORMED[@]} seed(s) formed; stop after the switch step"; exit 0; }

# --- audit + frozen sets per formed seed ---
for s in "${FORMED[@]}"; do
  [ -f "$OUTD/audit/s$s.json" ] && continue
  $PY audit_pick.py --seed $s > "$L/audit_s$s.log" 2>&1 || { log "FAILED audit s$s"; tail -20 "$L/audit_s$s.log"; exit 1; }
  log "audit s$s: $(tail -1 "$L/audit_s$s.log")"
done

# --- retraining, frozen order; budget = 2 x T_f rounded up to a multiple of the eval grid ---
heads() { field "$OUTD/audit/s$1.json" "','.join(d['sets']['$2'])"; }
budget() { field "$OUTD/runs/switch_s$1.json" "$EVAL * -(-2 * d['t_f'] // $EVAL)"; }
ALL=$(field "$OUTD/audit/s${FORMED[0]}.json" "[k for k in d['sets'] if k.startswith('all')][0]")
for arm in set "$ALL" top1 leave1; do
  for s in "${FORMED[@]}"; do run "${arm}_s$s" --mode clamp --seed $s --arm "$arm" --heads "$(heads $s $arm)" --steps "$(budget $s)"; done
done
for arm in random1 random2; do
  for s in "${FORMED[@]:0:2}"; do run "${arm}_s$s" --mode clamp --seed $s --arm "$arm" --heads "$(heads $s $arm)" --steps "$(budget $s)"; done
done
log "ALL CELLS DONE"
