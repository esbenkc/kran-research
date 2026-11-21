#!/bin/bash
# One laptop-queue box (labq) of the 8B run. Idempotent and resumable (head-clamp-retrain-8b's run.sh pattern): every
# finished cell (data/runs/<name>.json, data/whole/<name>.json, data/audit/<name>.json) is skipped; a run cut by the
# box deadline saved data/ckpt/<name>.pt (committed with the outputs) and the next box resumes it.
#   RUNS="ao_s0 ao_s1 ao_s2 l0_s0"  the training runs of this box, in order (status cells first, PREREG 'Compute')
#   WARM="0 1 2"                     also audit + whole-graph-score the warm adapters of these seeds (reported rows)
#   HOURS=4                          box length in AWAKE hours (macOS CLOCK_UPTIME_RAW; sleep does not count)
# Names: ao_s<k> = AO (ratio, weight 10); p100_s<k>, p10_s<k> = parent penalty at 100, 10; l0_s<k> = no penalty;
#   <name>_g2048 = fake-no guard: continue <name> (data/final_state/<name>.pt) to 2,048 updates.
# Per box: cache (stock torch.compile, read-only q_norm hook at block n_layers//2, 704 prompts; sha256 checked against
#   the first box and printed next to head-clamp-retrain-8b's) -> train every run -> whole-graph eval of every
#   trained run (one 8B load) -> head audit of every run. Stops at the deadline.
# labq --cmd: uv sync --frozen --no-install-project && RUNS="..." HOURS=4 caffeinate -i ./run.sh
# Smoke on the mini: AP_MODEL=Qwen/Qwen3-0.6B AP_OUT=smoke/box STEPS=16 EVAL=8 RUNS="ao_s0 l0_s0" ./run.sh
set -u
cd "$(dirname "$0")"
export AP_MODEL=${AP_MODEL:-Qwen/Qwen3-8B} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUTD=${AP_OUT:-data}; export AP_OUT=$OUTD
PY=${PY:-"uv run --frozen --no-sync python"}
HOURS=${HOURS:-4}; RUNS=${RUNS:-}; WARM=${WARM:-}
STEPS=${STEPS:-512}; GSTEPS=${GSTEPS:-2048}; EVAL=${EVAL:-64}; WARMADP=${WARMADP:-adapters_in}
JOB=${LABQ_JOB_ID:-local_$(date +%m%d_%H%M%S)}
L=$OUTD/laptop/$JOB; mkdir -p "$L" raw "$OUTD"
awake() { python3 -c "import time; print(int(time.clock_gettime(time.CLOCK_UPTIME_RAW)))"; }
START=$(date +%s); START_AW=$(awake); DEADLINE=$(python3 -c "print(int($START_AW + $HOURS * 3600 - 300))")
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
left() { echo $((DEADLINE - $(awake))); }
state() { echo "$(date '+%F %T') $JOB $*" > "$OUTD/run_state.txt"; }
SHORT=$(basename "$AP_MODEL"); CD=raw/step2/$SHORT/compile_C
trap 'rc=$?; log "box clock: awake $(( $(awake) - START_AW )) s, wall $(( $(date +%s) - START )) s rc $rc"; [ -d "$OUTD/adapters" ] && (cd "$OUTD/adapters" && shasum -a 256 *_adapter.pt > SHA256SUMS 2>/dev/null); [ $rc -ne 0 ] && state "FAIL rc $rc, see $L/chain.log"' EXIT
log "START box $JOB model $AP_MODEL commit $(git rev-parse --short HEAD 2>/dev/null) hours $HOURS runs [$RUNS] warm [$WARM] out $OUTD"

# --- cache ---
cp inputs/prompts.jsonl raw/prompts.jsonl
if [ ! -f "$CD/probe_rec_tail.npy" ]; then
  LAYER=$($PY -c "import cfgm; print(cfgm.PROBE_LAYER)" 2>/dev/null | tail -1)
  log "cache build: probe hook block $LAYER"
  /usr/bin/time -l $PY step1_torch.py --model "$AP_MODEL" --layer "$LAYER" --backend compile --tag C --split all --n -1 \
    --one-rep --passes rec --builds none,probe --out raw/step2 > "$L/cache.log" 2>&1 \
    || { log "FAILED cache build"; tail -20 "$L/cache.log"; exit 1; }
  log "cache built: $(grep -E 'none rec|probe rec' "$L/cache.log" | tr '\n' ' ')"
fi
SHA=$(cd "$CD" && shasum -a 256 none_rec_tail.npy probe_rec_tail.npy)
echo "$SHA" > "$L/cache_sha.txt"
if [ -f "$OUTD/cache_sha.txt" ]; then
  [ "$SHA" = "$(cat "$OUTD/cache_sha.txt")" ] || { log "ABORT: cache sha differs from the first box"; exit 1; }
  log "cache sha = first box"
else
  echo "$SHA" > "$OUTD/cache_sha.txt"; log "cache sha written (first box)"
fi
[ "$AP_MODEL" = Qwen/Qwen3-8B ] && { [ "$SHA" = "$(cat inputs/cache_sha_headclamp8b.txt)" ] && log "cache sha = head-clamp-retrain-8b's cache" || log "NOTE: cache sha differs from head-clamp-retrain-8b's cache (reported, not fatal)"; }
# dev+test subset sha (Vera compares her own streamed cache with it)
$PY -c "
import hashlib, json, numpy as np, cfgm
pid = json.load(open(cfgm.CDIR / 'meta.json'))['prompt_ids']
sp = {json.loads(l)['id']: json.loads(l)['split'] for l in open('raw/prompts.jsonl')}
rows = [i for i, p in enumerate(pid) if sp[p] in ('dev', 'test')]
for b in ('none', 'probe'):
    a = np.ascontiguousarray(np.load(cfgm.CDIR / f'{b}_rec_tail.npy', mmap_mode='r')[rows])
    print(b, 'dev+test rows', len(rows), hashlib.sha256(a.tobytes()).hexdigest())
" > "$L/cache_devtest_sha.txt" 2>/dev/null; log "dev+test cache sha: $(tr '\n' ' ' < "$L/cache_devtest_sha.txt")"

args_of() {  # training args of a run name
  local n=$1 base s
  base=${n%_g2048}; s=${base##*_s}
  case $base in
    ao_s*)   echo "--kind ratio --lam 10 --seed $s" ;;
    p100_s*) echo "--kind parent --lam 100 --seed $s" ;;
    p10_s*)  echo "--kind parent --lam 10 --seed $s" ;;
    l0_s*)   echo "--kind none --lam 0 --seed $s" ;;
    *) echo "BAD"; return 1 ;;
  esac
  if [ "$n" != "$base" ]; then echo "--steps $GSTEPS --continue-from $OUTD/final_state/$base.pt"
  else echo "--steps $STEPS"; fi
  case $base in ao_s0|p100_s0) echo "--keep-final" ;; esac
  [ "$WARMADP" != adapters_in ] && echo "--warm $WARMADP/switch_s${s}_adapter.pt"
}
seed_of() { local b=${1%_g2048}; echo "${b##*_s}"; }

# --- training ---
for n in $RUNS; do
  [ -f "$OUTD/runs/$n.json" ] && continue
  [ "$(left)" -lt 600 ] && { log "BOX END: not starting $n ($(left) s left)"; state "MORE train $n"; exit 0; }
  A=$(args_of "$n" | tr '\n' ' '); [[ "$A" == *BAD* ]] && { log "bad run name $n"; exit 1; }
  log "START $n :: $A"
  /usr/bin/time -l $PY train_pen.py --name "$n" $A --eval-every "$EVAL" --deadline "$DEADLINE" > "$L/$n.log" 2>&1
  rc=$?
  [ $rc -eq 3 ] && { log "BOX END: $n saved a checkpoint ($(grep DEADLINE "$L/$n.log"))"; state "MORE resume $n"; exit 0; }
  [ $rc -ne 0 ] && { log "FAILED $n (rc $rc)"; tail -20 "$L/$n.log"; exit 1; }
  log "OK $(grep '^DONE' "$L/$n.log" | tail -1) peak $(grep 'peak memory' "$L/$n.log" | awk '{printf "%.1f GB", $1/1e9}')"
done

# --- whole-graph eval (status switch) of every trained run of this box + the warm adapters ---
SPEC=""
for n in $RUNS; do [ -f "$OUTD/whole/$n.json" ] || SPEC="$SPEC,$n=$OUTD/adapters/${n}_adapter.pt"; done
for s in $WARM; do [ -f "$OUTD/whole/warm_s$s.json" ] || SPEC="$SPEC,warm_s$s=$WARMADP/switch_s${s}_adapter.pt"; done
SPEC=${SPEC#,}
if [ -n "$SPEC" ]; then
  [ "$(left)" -lt 900 ] && { log "BOX END: no time for the whole-graph eval"; state "MORE whole"; exit 0; }
  log "WHOLE $SPEC"
  /usr/bin/time -l $PY eval_whole.py --runs "$SPEC" > "$L/whole.log" 2>&1 || { log "FAILED eval_whole"; tail -20 "$L/whole.log"; exit 1; }
  grep '^WHOLE' "$L/whole.log" | while read -r line; do log "$line"; done
fi

# --- head audit (R_blind3 and the reported rows) ---
for n in $RUNS $(for s in $WARM; do echo warm_s$s; done); do
  [ -f "$OUTD/audit/$n.json" ] && continue
  [ "$(left)" -lt 1200 ] && { log "BOX END: not auditing $n ($(left) s left)"; state "MORE audit $n"; exit 0; }
  s=$(seed_of "$n")
  if [[ $n == warm_s* ]]; then ADP=$WARMADP/switch_s${s}_adapter.pt; REF=""; else ADP=$OUTD/adapters/${n}_adapter.pt; REF="--ref-adapter $WARMADP/switch_s${s}_adapter.pt"; fi
  log "AUDIT $n"
  /usr/bin/time -l $PY audit.py --run "$n" --adapter "$ADP" $REF > "$L/audit_$n.log" 2>&1 || { log "FAILED audit $n"; tail -20 "$L/audit_$n.log"; exit 1; }
  log "OK audit $n $(grep -o '"title_cell_blind_top3_removed": [-0-9.e]*' "$L/audit_$n.log") secs $(grep -o 'secs [0-9.]*' "$L/audit_$n.log" | tail -1)"
done
[ -f status.py ] && $PY status.py > "$L/status.log" 2>&1 && log "status: $(tail -1 "$L/status.log")"
log "BOX DONE"; state "DONE runs [$RUNS] warm [$WARM]"
