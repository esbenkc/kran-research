#!/bin/bash
# One laptop-queue box (labq) of the 8B run. Idempotent and resumable: every finished cell (data/runs/<name>.json,
# data/whole/<name>.json, data/audit/<name>.json, ...) is skipped; a run cut by the box deadline saved ckpt/<name>/ and
# the next box resumes it bit for bit. The labq --cmd fetches the branch head first (every earlier box's outputs), then
#   uv sync --frozen --no-install-project && HOURS=<h> caffeinate -i ./run.sh
# Order (PREREG "Budgets, early stop, order" + DEVIATIONS D1): cache (+ sha check vs the first box) -> G0 -> base maps
#   -> base whole-graph score -> ALL-8 s0 -> [PREDICTIONS.md section B on the branch head] -> last-MLP s0, s1 -> MLP-8 s0
#   (-> MARK-8 s0 if null) -> MLP-8 s1 -> ALL-8 s1 -> MLP-8 s2 -> audits of the MLP-8 models (+ after-map, attention
#   rows) -> ALL-8 s2 -> budget extensions of null MLP-8 runs -> ALL-8 s0 audit. Whole-graph score right after each run.
# STOP_AFTER=map ends the box after the base maps (J0a).
# Smoke on the mini: PS_MODEL=Qwen/Qwen3-0.6B PS_OUT=smoke/q06 CAP=64 FLOOR=64 LASTSTEPS=32 EVAL=32 HOURS=0.5 ./run.sh
set -u
cd "$(dirname "$0")"
export PS_MODEL=${PS_MODEL:-Qwen/Qwen3-8B} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
SK="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"  # sklearn steps (Felix #24426: libomp hang)
OUTD=${PS_OUT:-.}; export PS_OUT=$OUTD
D=$OUTD/data; C=$OUTD/raw/cache
PY=${PY:-"uv run --frozen --no-sync python"}
HOURS=${HOURS:-1}
EVAL=${EVAL:-128}; CAP=${CAP:-8192}; FLOOR=${FLOOR:-2048}; LASTSTEPS=${LASTSTEPS:-2048}
STOP_AFTER=${STOP_AFTER:-}
BRANCH=${BRANCH:-swarm/oracle-1118-frozen-heads-8b}
JOB=${LABQ_JOB_ID:-local_$(date +%m%d_%H%M%S)}
L=$D/logs/$JOB; mkdir -p "$L" "$D/runs" "$D/whole" "$OUTD/research_checks"
awake() { python3 -c "import time; print(int(time.clock_gettime(time.CLOCK_UPTIME_RAW)))"; }
START=$(date +%s); START_AW=$(awake); DEADLINE=$(python3 -c "print(int($START_AW + $HOURS * 3600 - 300))")
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
left() { echo $((DEADLINE - $(awake))); }
state() { echo "$(date '+%F %T') $*" > "$D/run_state.txt"; }
field() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
trap 'rc=$?; log "box clock: awake $(( $(awake) - START_AW )) s, wall $(( $(date +%s) - START )) s, rc $rc"; [ -d "$OUTD/adapters" ] && (cd "$OUTD/adapters" && shasum -a 256 *_adapter.pt > SHA256SUMS 2>/dev/null); [ $rc -ne 0 ] && state "FAIL rc $rc, see $L/chain.log"' EXIT
log "START box $JOB model $PS_MODEL commit $(git rev-parse --short HEAD 2>/dev/null) hours $HOURS out $OUTD"

# --- cache: whole compiled graph, read-only q_norm hook at block n_layers//2, one recorder at the block-FIRST input ---
if [ ! -f "$C/probe_A_tailin.npy" ]; then
  log "cache build (pass A)"
  /usr/bin/time -l $PY cache.py --pass A > "$L/cache_A.log" 2>&1 || { log "FAILED cache A"; tail -20 "$L/cache_A.log"; exit 1; }
  log "cache built: $(grep -E '^(none|probe) A' "$L/cache_A.log" | tr '\n' ' ') peak $(grep 'peak memory footprint' "$L/cache_A.log" | awk '{print $1}')"
fi
SHA=$(cd "$C" && shasum -a 256 none_A_tailin.npy probe_A_tailin.npy none_A_logits.npy probe_A_logits.npy)
echo "$SHA" > "$L/cache_sha.txt"  # this box's raw bf16 bits of both conditions (Vera #24216)
if [ -f "$D/cache_sha.txt" ]; then
  [ "$SHA" = "$(cat "$D/cache_sha.txt")" ] || { log "ABORT: cache sha differs from the first box"; echo "$SHA" > "$L/cache_sha_mismatch.txt"; exit 1; }
  log "cache sha = first box"
else
  echo "$SHA" > "$D/cache_sha.txt"; cp "$C/meta_A.json" "$D/cache_meta_A.json"; log "cache sha written (first box)"
fi

# --- G0: does the probe change the block-FIRST input? (share of prompts with any differing bit, per position group) ---
if [ ! -f "$D/g0.json" ]; then
  $PY g0.py > "$L/g0.log" 2>&1 || { log "FAILED g0"; tail -20 "$L/g0.log"; exit 1; }
  log "G0: $(tail -1 "$L/g0.log")"
fi
G0=$(field "$D/g0.json" "d['by_split']['test']['share_any_pos_differs']")
python3 -c "import sys; sys.exit(0 if $G0 >= 0.95 else 1)" || { log "G0 NULL (PREDICTIONS.md A1): test share with any differing bit $G0 < 0.95; no training"; state "STOP G0 null $G0"; exit 0; }

# --- base maps (untrained model, eager tail on the cached inputs) + the last-MLP inputs ---
LASTL=$($PY -c "import cfgm; print(cfgm.LAST)" 2>/dev/null | tail -1)
need_states() { [ ! -f "$OUTD/research_checks/map.json" ] || { [ ! -f "$C/probe_eager_mid${LASTL}_79.npy" ] && { [ ! -f "$D/runs/mlplast_none_s0.json" ] || [ ! -f "$D/runs/mlplast_none_s1.json" ]; }; }; }
if need_states; then
  log "map states"
  $PY research_checks/map_states.py states . "$OUTD/raw/map" > "$L/map_states.log" 2>&1 || { log "FAILED map states"; tail -20 "$L/map_states.log"; exit 1; }
  MSHA=$(cd "$C" && shasum -a 256 none_eager_mid${LASTL}_79.npy probe_eager_mid${LASTL}_79.npy)
  if [ -f "$D/lastmlp_inputs_sha.txt" ]; then
    [ "$MSHA" = "$(cat "$D/lastmlp_inputs_sha.txt")" ] || { log "ABORT: last-MLP inputs differ from the first box"; exit 1; }
  else
    echo "$MSHA" > "$D/lastmlp_inputs_sha.txt"
  fi
  cp "$OUTD/raw/map/eager_meta.json" "$OUTD/research_checks/eager_meta.json"
fi
if [ ! -f "$OUTD/research_checks/map.json" ]; then
  $SK $PY research_checks/map_probe.py . "$OUTD/raw/map" "$OUTD/research_checks/map.json" > "$L/map_probe.log" 2>&1 || { log "FAILED map probe"; tail -20 "$L/map_probe.log"; exit 1; }
  log "map: $(tail -1 "$L/map_probe.log" | cut -c1-600)"
fi
if [ ! -f "$OUTD/research_checks/route.json" ]; then
  $PY research_checks/map_states.py route . "$OUTD/research_checks" > "$L/route.log" 2>&1 || { log "FAILED route"; tail -20 "$L/route.log"; exit 1; }
  log "route: $(tail -1 "$L/route.log" | cut -c1-600)"
fi
if [ ! -f "$OUTD/research_checks/after_attn/attn_base_test.json" ]; then
  $PY research_checks/map_states.py attn . "$OUTD/research_checks/after_attn" --adapter none --mapjson "$OUTD/research_checks/map.json" > "$L/attn_base.log" 2>&1 || { log "FAILED attn base"; tail -20 "$L/attn_base.log"; exit 1; }
fi
if [ ! -f "$D/skeptic_pack/meta.json" ] && [ -f "$C/probe_eager_mid${LASTL}_79.npy" ]; then
  $PY skeptic_pack.py > "$L/skeptic_pack.log" 2>&1 && log "$(tail -1 "$L/skeptic_pack.log")"  # Vera #24591
fi
$PY predict.py > "$L/predict.log" 2>&1 && log "map prediction: $(tail -1 "$L/predict.log")"
if [ "$STOP_AFTER" = map ]; then log "STOP_AFTER=map"; state "MORE after map"; exit 0; fi

# --- whole-graph title score (base model once, then each trained model right after its run) ---
whole() {
  [ -f "$D/whole/$1.json" ] && return 0
  [ "$(left)" -lt 900 ] && { log "BOX END: not scoring $1 ($(left) s left)"; state "MORE whole $1"; exit 0; }
  /usr/bin/time -l $PY eval_whole.py --runs "$1" ${2:-} > "$L/whole_$1.log" 2>&1 || { log "FAILED whole $1"; tail -20 "$L/whole_$1.log"; exit 1; }
  log "whole $1: strict $(field "$D/whole/$1.json" "d['strict']") repeat-changed $(field "$D/whole/$1.json" "d['info']") peak $(grep 'peak memory footprint' "$L/whole_$1.log" | awk '{print $1}')"
}
whole base

# --- training: run <name> <train.py args>: skip if done (and long enough); stop the box near the deadline ---
run() {
  local n=$1 steps=$2; shift 2
  if [ -f "$D/runs/$n.json" ]; then
    [ -z "${EXTEND:-}" ] && return 0
    local done_steps=$(field "$D/runs/$n.json" "d['steps_done']") es=$(field "$D/runs/$n.json" "d['early_stop_step']")
    { [ "$es" != None ] || [ "$done_steps" -ge "$steps" ]; } && return 0
    log "EXTEND $n from $done_steps to $steps updates (PREREG budget rule: null run, later same-seed ALL-8 T75)"
    rm -f "$D/whole/$n.json"
  fi
  [ "$(left)" -lt 600 ] && { log "BOX END: not starting $n ($(left) s left)"; state "MORE next $n"; exit 0; }
  log "START $n :: --steps $steps $*"
  $PY train.py --steps "$steps" "$@" --eval-every "$EVAL" --deadline "$DEADLINE" > "$L/$n.log" 2>&1
  local rc=$?
  [ $rc -eq 3 ] && { log "BOX END: $n saved a checkpoint at the deadline ($(grep DEADLINE "$L/$n.log"))"; state "MORE resume $n"; exit 0; }
  [ $rc -ne 0 ] && { log "FAILED $n (rc $rc)"; tail -20 "$L/$n.log"; exit 1; }
  log "OK $n: steps $(field "$D/runs/$n.json" "d['steps_done']") early stop $(field "$D/runs/$n.json" "d['early_stop_step']") T75 $(field "$D/runs/$n.json" "d['T75']") cached test strict $(field "$D/runs/$n.json" "d['test_cached']['strict']") train_s $(field "$D/runs/$n.json" "d['train_s']") ($(field "$D/runs/$n.json" "round(d['train_s'] / max(1, d['steps_done'] - (d['resumed_at'][0] if d['resumed_at'] else 0)), 2)") s/upd this segment)"
}
t75() { [ -f "$D/runs/all8_none_s$1.json" ] && field "$D/runs/all8_none_s$1.json" "d['T75'] or 0" || echo 0; }
mlpbudget() {  # PREREG: max(FLOOR, 2 x T75 of ALL-8 s0, 2 x T75 of same-seed ALL-8 if it ran), cap CAP
  python3 -c "print(min($CAP, max($FLOOR, 2 * $(t75 0), 2 * $(t75 $1))))"
}
audit() {
  local n=$1
  if [ ! -f "$D/audit/$n.json" ]; then
    [ "$(left)" -lt 1500 ] && { log "BOX END: not auditing $n ($(left) s left)"; state "MORE audit $n"; exit 0; }
    $PY audit.py --run "$n" > "$L/audit_$n.log" 2>&1 || { log "FAILED audit $n"; tail -20 "$L/audit_$n.log"; exit 1; }
    log "audit $n: $(grep -E '^R_blind3|^clamp_all' "$L/audit_$n.log" | cut -c1-300 | tr '\n' ' ')"
  fi
  if [ ! -f "$D/after_map/$n.json" ] && [ -f "$OUTD/raw/after/${n}_states.npy" ]; then
    $SK $PY after_map.py --run "$n" > "$L/after_map_$n.log" 2>&1 || { log "FAILED after_map $n"; tail -20 "$L/after_map_$n.log"; exit 1; }
  fi
  if [ ! -f "$OUTD/research_checks/after_attn/attn_${n}_adapter_test.json" ]; then
    $PY research_checks/map_states.py attn . "$OUTD/research_checks/after_attn" --adapter "$OUTD/adapters/${n}_adapter.pt" --mapjson "$OUTD/research_checks/map.json" > "$L/attn_$n.log" 2>&1 || { log "FAILED attn $n"; tail -20 "$L/attn_$n.log"; exit 1; }
    log "attn $n: $(grep -m1 top10 "$L/attn_$n.log" | cut -c1-300)"
  fi
}

run all8_none_s0 "$CAP" --arm all8 --seed 0; whole all8_none_s0

# --- every MLP run waits for PREDICTIONS.md section B (map values) on the branch head: the marker line below ---
MARK_B='<!-- SECTION B FROZEN -->'
if ! { [ -f PREDICTIONS.md ] && grep -qx "$MARK_B" PREDICTIONS.md; } && [ -z "${SMOKE_NO_GATE:-}" ]; then
  git fetch -q origin "$BRANCH" 2>/dev/null
  if git show "FETCH_HEAD:projects/probe-switch-frozen-heads-8b/PREDICTIONS.md" 2>/dev/null | grep -qx "$MARK_B"; then
    git show "FETCH_HEAD:projects/probe-switch-frozen-heads-8b/PREDICTIONS.md" > PREDICTIONS.md
    log "PREDICTIONS.md section B found on the branch head $(git rev-parse --short FETCH_HEAD)"
  else
    log "GATE: PREDICTIONS.md section B is not on $BRANCH yet; no MLP run in this box"; state "WAIT predictions"; exit 0
  fi
fi

run mlplast_none_s0 "$LASTSTEPS" --arm mlplast --seed 0; whole mlplast_none_s0
run mlplast_none_s1 "$LASTSTEPS" --arm mlplast --seed 1; whole mlplast_none_s1
run mlp8_none_s0 "$(mlpbudget 0)" --arm mlp8 --seed 0; whole mlp8_none_s0
S0=$(field "$D/whole/mlp8_none_s0.json" "d['strict']")
if python3 -c "import sys; sys.exit(0 if $S0 < 0.25 else 1)"; then
  log "MLP-8 s0 null ($S0): MARK-8 s0 next (PREREG)"
  if [ ! -f "$C/probe_M_tailin.npy" ]; then
    $PY cache.py --pass M > "$L/cache_M.log" 2>&1 || { log "FAILED cache M"; tail -20 "$L/cache_M.log"; exit 1; }
  fi
  run mlp8_none_s0_mark "$(mlpbudget 0)" --arm mlp8 --seed 0 --input M
  whole mlp8_none_s0_mark --mark
fi
# order (DEVIATIONS D1, Tove #24219): the status cells first, then the audits, then ALL-8 s2 and the extensions
run mlp8_none_s1 "$(mlpbudget 1)" --arm mlp8 --seed 1; whole mlp8_none_s1
run all8_none_s1 "$CAP" --arm all8 --seed 1; whole all8_none_s1
run mlp8_none_s2 "$(mlpbudget 2)" --arm mlp8 --seed 2; whole mlp8_none_s2
$PY summary.py > "$L/summary.log" 2>&1 && log "summary: $(tail -1 "$L/summary.log" | cut -c1-400)"
audit mlp8_none_s0; audit mlp8_none_s1; audit mlp8_none_s2
run all8_none_s2 "$CAP" --arm all8 --seed 2; whole all8_none_s2
# budget extensions: a null MLP-8 seed whose same-seed ALL-8 T75 came later (PREREG)
for s in 0 1 2; do
  n=mlp8_none_s$s
  st=$(field "$D/whole/$n.json" "d['strict']")
  python3 -c "import sys; sys.exit(0 if $st < 0.25 else 1)" && { EXTEND=1 run $n "$(mlpbudget $s)" --arm mlp8 --seed $s; whole $n; }
done
audit all8_none_s0
$PY summary.py > "$L/summary.log" 2>&1 && log "summary: $(tail -1 "$L/summary.log" | cut -c1-400)"
log "ALL CELLS DONE"; state "DONE all cells"
