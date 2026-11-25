#!/bin/bash
# One laptop-queue box (labq) of probe-switch-head-budget, Qwen3-1.7B. Idempotent and resumable: every finished cell
# (data/runs/<n>.json, data/whole/<n>.json, data/kaudit/<n>.json) is skipped; a run cut by the box deadline saved
# data/ckpt/<n>.pt and the next box resumes it.
#   ARMS="6 3 1"   attacker budgets k_att, in this order (PREREG: status arm 6 first). Per arm: train k<A>_s0..s2
#                  (warm = parent17 switch_s<seed>), fade check, the spare k<A>_s3 (warm s0, training seed 3) when a
#                  seed faded (cached-tail fp32 strict switch < 0.75 after training; max 1 spare per arm), whole-graph
#                  eval of the arm, then the k-sweep audit (kaudit.py) of every run of the arm.
#   L0="0 1"       control L0: same training, no penalty (seeds), trained after the arms, whole-graph eval + kaudit.
#   CONTROLS=1     after the arms: kaudit of the untrained controls warm_s0..s2 (W = the parent17 switch adapters).
#   PARENT=""      side row: kaudit of the parent's saved k_att=3 adapters (inputs/parent17_ao/ao_s<s>_adapter.pt).
#   J1 (status):  ARMS="6" L0="0 1" CONTROLS=1          J2 (law points): ARMS="3 1" L0="" CONTROLS=0 PARENT="0 1 2"
#   HOURS=4        box length in AWAKE hours (macOS CLOCK_UPTIME_RAW; sleep does not count).
#   REPICK=8       the attacker's monitor re-picks its joint-greedy top-k_att every REPICK updates (PREREG).
# labq --cmd: uv sync --frozen --no-install-project && ARMS="6 3 1" HOURS=4 caffeinate -i ./run_k.sh
# Smoke on the mini: AP_MODEL=Qwen/Qwen3-0.6B AP_OUT=smoke/box STEPS=16 EVAL=8 MINWARM=0 WARMADP=smoke/adp06 \
#   ARMS="6" SEEDS="0" CONTROLS=0 KMAX=4 NRAND=3 ./run_k.sh
set -u
cd "$(dirname "$0")"
export AP_MODEL=${AP_MODEL:-Qwen/Qwen3-1.7B} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUTD=${AP_OUT:-data}; export AP_OUT=$OUTD
PY=${PY:-"uv run --frozen --no-sync python"}
HOURS=${HOURS:-4}; ARMS=${ARMS:-"6 3 1"}; SEEDS=${SEEDS:-"0 1 2"}; CONTROLS=${CONTROLS:-1}; L0=${L0-"0 1"}
PARENT=${PARENT:-}; MICRO=${MICRO:-4}
STEPS=${STEPS:-512}; EVAL=${EVAL:-64}; WARMADP=${WARMADP:-inputs/parent17}; MINWARM=${MINWARM:-0.75}
REPICK=${REPICK:-8}; MONN=${MONN:-64}; KMAX=${KMAX:-12}; NRAND=${NRAND:-50}; TEMPS=${TEMPS:-B,16,4,1}
JOB=${LABQ_JOB_ID:-local_$(date +%m%d_%H%M%S)}
L=$OUTD/laptop/$JOB; mkdir -p "$L" raw "$OUTD"
awake() { python3 -c "import time; print(int(time.clock_gettime(time.CLOCK_UPTIME_RAW)))"; }
START=$(date +%s); START_AW=$(awake); DEADLINE=$(python3 -c "print(int($START_AW + $HOURS * 3600 - 300))")
# STOP_AT (unix seconds, wall clock): the box stops by itself before it (PREREG D3: a cancelled labq job commits
# nothing, a job that exits does), so finished seeds land on the branch. train_pen gets the earlier of both deadlines.
STOP_AT=${STOP_AT:-0}
if [ "$STOP_AT" -gt 0 ]; then
  D2=$((START_AW + STOP_AT - START - 120)); [ "$D2" -lt "$DEADLINE" ] && DEADLINE=$D2
fi
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
left() { local a=$((DEADLINE - $(awake))) w; [ "$STOP_AT" -gt 0 ] && { w=$((STOP_AT - 120 - $(date +%s))); [ $w -lt $a ] && a=$w; }; echo $a; }
state() { echo "$(date '+%F %T') $JOB $*" > "$L/run_state.txt"; }
SHORT=$(basename "$AP_MODEL"); CD=raw/step2/$SHORT/compile_C
trap 'rc=$?; log "box clock: awake $(( $(awake) - START_AW )) s, wall $(( $(date +%s) - START )) s rc $rc"; [ -d "$OUTD/adapters" ] && (cd "$OUTD/adapters" && shasum -a 256 *_adapter.pt) > "$L/adapters_SHA256SUMS.txt" 2>/dev/null; [ $rc -ne 0 ] && state "FAIL rc $rc, see $L/chain.log"' EXIT
log "START box $JOB model $AP_MODEL commit $(git rev-parse --short HEAD 2>/dev/null) hours $HOURS arms [$ARMS] seeds [$SEEDS] l0 [$L0] controls $CONTROLS parent [$PARENT] out $OUTD"

# --- cache (the parent's compiled cache recipe; its sha256 is checked against the parent's 1.7B cache) ---
cp inputs/prompts.jsonl raw/prompts.jsonl
LAYER=$($PY -c "import cfgm; print(cfgm.PROBE_LAYER)")
if [ ! -f "$CD/probe_rec_tail.npy" ]; then
  log "cache build: probe hook block $LAYER"
  /usr/bin/time -l $PY step1_torch.py --model "$AP_MODEL" --layer "$LAYER" --backend compile --tag C --split all --n -1 \
    --one-rep --passes rec --builds none,probe --out raw/step2 > "$L/cache.log" 2>&1 \
    || { log "FAILED cache build"; tail -20 "$L/cache.log"; exit 1; }
  log "cache built: $(grep -E 'none rec|probe rec' "$L/cache.log" | tr '\n' ' ')"
fi
SHA=$(cd "$CD" && shasum -a 256 none_rec_tail.npy probe_rec_tail.npy)
echo "$SHA" > "$L/cache_sha.txt"
[ "$AP_MODEL" = Qwen/Qwen3-1.7B ] && { [ "$SHA" = "$(cat inputs/cache_sha_parent17.txt)" ] && log "cache sha = the parent's 1.7B cache (probe-switch-audit-penalty-8b data1.7b)" || log "NOTE: cache sha differs from the parent's 1.7B cache (reported, not fatal; the warm gate decides)"; }

args_of() {  # training args of a run name k<A>_s<s> or l0_s<s>; the spare s3 starts from warm s0 with training seed 3
  local n=$1 A s w
  s=${n##*_s}; w=$s; [ "$s" = 3 ] && w=0
  if [[ $n == l0_s* ]]; then echo "--kind none --lam 0 --fp32-logits"
  else A=${n%%_s*}; A=${A#k}
       echo "--kind ratio --lam 10 --katt $A --temp B --repick $REPICK --mon train --mon-n $MONN --fp32-logits"; fi
  echo "--seed $s --steps $STEPS --micro $MICRO --warm $WARMADP/switch_s${w}_adapter.pt"
}
train() {
  local n=$1
  [ -f "$OUTD/runs/$n.json" ] && return 0
  [ "$(left)" -lt 900 ] && { log "BOX END: not starting $n ($(left) s left)"; state "MORE train $n"; exit 0; }
  local TA
  TA=$(args_of "$n" | tr '\n' ' ')
  log "START $n :: $TA"
  /usr/bin/time -l $PY train_pen.py --name "$n" $TA --eval-every "$EVAL" --deadline "$DEADLINE" --min-warm "$MINWARM" > "$L/$n.log" 2>&1
  rc=$?
  [ $rc -eq 4 ] && { log "$(grep 'WARM GATE FAIL' "$L/$n.log"): the cache does not carry the trigger; stop the box"; state "STOP warm gate $n"; exit 0; }
  [ $rc -eq 3 ] && { log "BOX END: $n saved a checkpoint ($(grep DEADLINE "$L/$n.log"))"; state "MORE resume $n"; exit 0; }
  [ $rc -ne 0 ] && { log "FAILED $n (rc $rc)"; tail -20 "$L/$n.log"; exit 1; }
  log "OK $(grep '^DONE' "$L/$n.log" | tail -1) peak $(grep 'peak memory' "$L/$n.log" | awk '{printf "%.1f GB", $1/1e9}')"
}
faded() {  # 1 when the run's cached-tail fp32 strict switch after training is below 0.75 (PREREG 'live' gate)
  python3 -c "import json,sys; r=json.load(open('$OUTD/runs/$1.json')); print(int(r['test_cached']['switch_strict32'] < 0.75))"
}
whole() {
  local SPEC=""
  for n in "$@"; do
    adp=$OUTD/adapters/${n}_adapter.pt; [[ $n == warm_s* ]] && adp=$WARMADP/switch_s${n#warm_s}_adapter.pt
    [[ $n == parent_ao_s* ]] && adp=inputs/parent17_ao/ao_s${n#parent_ao_s}_adapter.pt
    [ -f "$OUTD/whole/$n.json" ] || SPEC="$SPEC,$n=$adp"
  done
  SPEC=${SPEC#,}
  [ -z "$SPEC" ] && return 0
  [ "$(left)" -lt 600 ] && { log "BOX END: no time for the whole-graph eval"; state "MORE whole"; exit 0; }
  log "WHOLE $SPEC"
  /usr/bin/time -l $PY eval_whole.py --runs "$SPEC" > "$L/whole_$(date +%H%M%S).log" 2>&1 || { log "FAILED eval_whole"; exit 1; }
  grep -h '^WHOLE' "$L"/whole_*.log | sort -u | while read -r line; do log "$line"; done
}
kaud() {  # kaudit of one run; random-set sizes k_att+2 and 2 k_att (+8 for every arm, the status arm's size)
  local n=$1 A sizes adp
  [ -f "$OUTD/kaudit/$n.json" ] && return 0
  [ "$(left)" -lt 600 ] && { log "BOX END: not auditing $n ($(left) s left)"; state "MORE kaudit $n"; exit 0; }
  if [[ $n == warm_s* ]]; then adp=$WARMADP/switch_s${n#warm_s}_adapter.pt; sizes="8,12"
  elif [[ $n == parent_ao_s* ]]; then adp=inputs/parent17_ao/ao_s${n#parent_ao_s}_adapter.pt; sizes="5,6,8"
  elif [[ $n == l0_s* ]]; then adp=$OUTD/adapters/${n}_adapter.pt; sizes="8,12"
  else A=${n%%_s*}; A=${A#k}; adp=$OUTD/adapters/${n}_adapter.pt
       sizes=$(python3 -c "print(','.join(str(x) for x in sorted({$A + 2, 2 * $A, 8})))"); fi
  log "KAUDIT $n sizes $sizes"
  /usr/bin/time -l $PY kaudit.py --run "$n" --adapter "$adp" --temps "$TEMPS" --kmax "$KMAX" --n-random "$NRAND" \
    --rand-sizes "$sizes" > "$L/kaudit_$n.log" 2>&1 || { log "FAILED kaudit $n"; tail -20 "$L/kaudit_$n.log"; exit 1; }
  log "OK $(grep '^KAUDIT' "$L/kaudit_$n.log" | cut -c1-600) peak $(grep 'peak memory' "$L/kaudit_$n.log" | awk '{printf "%.1f GB", $1/1e9}')"
}

for A in $ARMS; do  # per seed: train, whole-graph eval, audit (a box that stops early keeps whole seeds)
  for s in $SEEDS; do train "k${A}_s$s"; whole "k${A}_s$s"; kaud "k${A}_s$s"; done
  NF=0; for s in $SEEDS; do NF=$((NF + $(faded "k${A}_s$s"))); done
  if [ "$NF" -ge 1 ]; then
    log "arm k_att=$A: $NF seed(s) faded (cached fp32 strict < 0.75) -> spare k${A}_s3"
    train "k${A}_s3"; whole "k${A}_s3"; kaud "k${A}_s3"
  fi
done
if [ -n "$L0" ]; then
  for s in $L0; do train "l0_s$s"; whole "l0_s$s"; kaud "l0_s$s"; done
fi
if [ "$CONTROLS" = 1 ]; then
  W=""; for s in $SEEDS; do W="$W warm_s$s"; done
  whole $W
  for n in $W; do kaud "$n"; done
fi
if [ -n "$PARENT" ]; then
  R=""; for s in $PARENT; do R="$R parent_ao_s$s"; done
  whole $R
  for n in $R; do kaud "$n"; done
fi
log "BOX DONE"; state "DONE arms [$ARMS] l0 [$L0] controls $CONTROLS parent [$PARENT]"
