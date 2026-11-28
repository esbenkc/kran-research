#!/bin/bash
# probe-switch-score-gaming: the whole run on the Mac mini, one job at a time (PREREG order):
#   t1_s0, t1_s1, t1_s2 (arm T1: forced attacker trained against the T = 1 blind audit, warm parent17 switch_s<s>),
#   then the readouts (single cuts + two-answer score) of the parent's adapters (inputs/), then t1_s3..t1_s5 if
#   PREREG's time rule allows them (training seed s, warm switch_s<s-3>).
# Each T1 run: train (train_pen.py), whole-graph check G1 (eval_whole.py), status audit (kaudit.py, status T = 1
# first), readouts (readouts.py), then commit its data/ files on the room branch. A finished step (its output JSON
# exists) is skipped, so the box can be restarted.
# Wall stop (PREREG): no training starts after 2026-11-28 04:35 CET (unix 1795836900); audits until 05:55 CET
# (1795841700).
#   caffeinate -i ./run_t1.sh [step ...]   (default: t1_s0 t1_s1 t1_s2 P t1_s3 t1_s4 t1_s5)
#   steps: t1_s<s> = one T1 cell; P = carrier sweep of the parent's 8 adapters; X = two-answer readouts (after 05:55)
#   spare (PREREG: training seed 6, warm of the failed seed): WARM6=<warm seed> ./run_t1.sh t1_s6
cd "$(dirname "$0")"
export AP_MODEL=${AP_MODEL:-Qwen/Qwen3-1.7B} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUTD=${AP_OUT:-data}; export AP_OUT=$OUTD
PY=${PY:-".venv/bin/python"}
BRANCH=swarm/oracle-1128-score-gaming-law
STEPS=${STEPS:-512}; EVAL=${EVAL:-64}; WARMADP=inputs/parent17; MINWARM=0.75
REPICK=8; MONN=64; KMAX=${KMAX:-12}; NRAND=${NRAND:-50}; TEMPS=${TEMPS:-1,B,16}; RTEMPS=${RTEMPS:-1,B,16,4}
FLOOR="--floor 0.97 --floor-eta 1.0 --mu-max 100"
FASTPICK=${FASTPICK:---fast-pick}
STOP_TRAIN=${STOP_TRAIN:-1795836900}; STOP_AUDIT=${STOP_AUDIT:-1795841700}
COMMIT=${COMMIT:-1}
JOB=mini_$(date +%m%d_%H%M%S)
L=$OUTD/mini/$JOB; mkdir -p "$L"
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
now() { date +%s; }
trap 'rc=$?; log "box end rc $rc"; [ -d "$OUTD/adapters" ] && (cd "$OUTD/adapters" && shasum -a 256 *_adapter.pt) > "$L/adapters_SHA256SUMS.txt" 2>/dev/null' EXIT
log "START box $JOB model $AP_MODEL commit $(git rev-parse --short HEAD) out $OUTD fastpick '$FASTPICK'"

CD=raw/step2/$(basename "$AP_MODEL")/compile_C
SHA=$(cd "$CD" && shasum -a 256 none_rec_tail.npy probe_rec_tail.npy) || { log "FAILED: no cache in $CD"; exit 1; }
echo "$SHA" > "$L/cache_sha.txt"
[ "$SHA" = "$(cat inputs/cache_sha_parent17.txt)" ] && log "cache sha = the parent's 1.7B cache" || { log "FAILED: cache sha differs"; exit 1; }

commit_twoans() {  # PREREG reported rows: the two-answer readout, audit-only, after 05:55 CET when the box is idle
  local n=$1
  [ -f "$OUTD/readouts/${n}_two_answer.json" ] && return 0
  /usr/bin/time -l $PY readouts.py --run "$n" --adapter "$(adapter_of "$n")" --temps "$RTEMPS" --kmax "$KMAX" --two-answer \
    > "$L/two_answer_$n.log" 2>&1 || { log "FAILED two-answer $n"; tail -20 "$L/two_answer_$n.log"; return 1; }
  log "OK $(grep '^READOUTS' "$L/two_answer_$n.log" | cut -c1-500)"
  [ "$COMMIT" = 1 ] && ( git add -- "$OUTD/readouts/${n}_two_answer.json" "$L" && git commit -q -m "probe-switch-score-gaming: two-answer readout $n" -- "$OUTD/readouts/${n}_two_answer.json" "$L" \
    && for i in 1 2 3; do git pull -q --rebase --autostash origin $BRANCH && git push -q origin HEAD:$BRANCH && break; sleep 5; done ) >> "$L/git.log" 2>&1
}
cell() {  # commit this run's data files on the room branch (only these paths), push; never fatal
  [ "$COMMIT" = 1 ] || return 0
  local n=$1 paths
  paths=$(ls -d "$OUTD/runs/$n.json" "$OUTD/whole/$n.json" "$OUTD/kaudit/$n.json" "$OUTD/readouts/$n.json" \
          "$OUTD/adapters/${n}_adapter.pt" "$L" 2>/dev/null)
  ( git add -- $paths && git commit -q -m "probe-switch-score-gaming: data $n (mini box $JOB)" -- $paths \
    && for i in 1 2 3; do git pull -q --rebase --autostash origin $BRANCH && git push -q origin HEAD:$BRANCH && break; sleep 5; done ) >> "$L/git.log" 2>&1 \
    && log "COMMIT $n $(git rev-parse --short HEAD)" || log "NOTE: commit/push of $n failed (see $L/git.log)"
}
warm_of() { local s=${1##*_s}; [ "$s" = 6 ] && { echo "${WARM6:?spare needs WARM6}"; return; }; echo $(( s % 3 )); }
adapter_of() {  # trained T1 runs: data/adapters; parent adapters: inputs/
  case $1 in
    t1_s*) echo "$OUTD/adapters/${1}_adapter.pt" ;;
    k6_s*) echo "inputs/parent_k6/${1}_adapter.pt" ;;
    l0_s*) echo "inputs/parent_l0floor/${1}_adapter.pt" ;;
    warm_s*) echo "$WARMADP/switch_s${1#warm_s}_adapter.pt" ;;
  esac
}
train() {
  local n=$1 s TA rc
  [ -f "$OUTD/runs/$n.json" ] && return 0
  [ "$(now)" -ge "$STOP_TRAIN" ] && { log "WALL STOP: not starting $n (no training after 04:35 CET)"; return 1; }
  s=${n##*_s}
  TA="--kind ratio --lam 10 --katt 6 --temp 1 --repick $REPICK --mon train --mon-n $MONN --fp32-logits $FLOOR $FASTPICK"
  TA="$TA --seed $s --steps $STEPS --micro 8 --warm $WARMADP/switch_s$(warm_of "$n")_adapter.pt"
  log "START $n :: $TA"
  /usr/bin/time -l $PY train_pen.py --name "$n" $TA --eval-every "$EVAL" --min-warm "$MINWARM" > "$L/$n.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { log "FAILED $n (rc $rc)"; tail -20 "$L/$n.log"; return 1; }
  log "OK $(grep '^DONE' "$L/$n.log" | tail -1) | $(grep '^TRAIN_FINAL' "$L/$n.log" | tail -1) | peak $(grep 'peak memory' "$L/$n.log" | awk '{printf "%.2f GB", $1/1e9}')"
}
whole() {
  local n=$1
  [ -f "$OUTD/whole/$n.json" ] && return 0
  [ "$(now)" -ge "$STOP_AUDIT" ] && { log "AUDIT STOP: no whole-graph check of $n"; return 1; }
  /usr/bin/time -l $PY eval_whole.py --runs "$n=$(adapter_of "$n")" > "$L/whole_$n.log" 2>&1 || { log "FAILED eval_whole $n"; tail -20 "$L/whole_$n.log"; return 1; }
  log "$(grep -h '^WHOLE' "$L/whole_$n.log" | tail -1) | peak $(grep 'peak memory' "$L/whole_$n.log" | awk '{printf "%.2f GB", $1/1e9}')"
}
kaud() {  # status audit (status T = 1 first); random-set sizes 8 (status) and 12 (reported)
  local n=$1
  [ -f "$OUTD/kaudit/$n.json" ] && return 0
  [ "$(now)" -ge "$STOP_AUDIT" ] && { log "AUDIT STOP: no audit of $n"; return 1; }
  log "KAUDIT $n"
  /usr/bin/time -l $PY kaudit.py --run "$n" --adapter "$(adapter_of "$n")" --temps "$TEMPS" --kmax "$KMAX" --n-random "$NRAND" \
    --rand-sizes 8,12 > "$L/kaudit_$n.log" 2>&1 || { log "FAILED kaudit $n"; tail -20 "$L/kaudit_$n.log"; return 1; }
  log "OK $(grep '^KAUDIT' "$L/kaudit_$n.log" | cut -c1-500) | peak $(grep 'peak memory' "$L/kaudit_$n.log" | awk '{printf "%.2f GB", $1/1e9}')"
}
reads() {  # readouts: single cuts of every head (oracle carrier), two-answer greedy
  local n=$1
  [ -f "$OUTD/readouts/$n.json" ] && return 0
  [ "$(now)" -ge "$STOP_AUDIT" ] && { log "AUDIT STOP: no readouts of $n"; return 1; }
  /usr/bin/time -l $PY readouts.py --run "$n" --adapter "$(adapter_of "$n")" --temps "$RTEMPS" --kmax "$KMAX" \
    > "$L/readouts_$n.log" 2>&1 || { log "FAILED readouts $n"; tail -20 "$L/readouts_$n.log"; return 1; }
  log "OK $(grep '^READOUTS' "$L/readouts_$n.log" | cut -c1-500) | peak $(grep 'peak memory' "$L/readouts_$n.log" | awk '{printf "%.2f GB", $1/1e9}')"
}
twoans() {  # PREREG reported rows: the two-answer readout, audit-only, after 05:55 CET when the box is idle
  local n=$1
  [ -f "$OUTD/readouts/${n}_two_answer.json" ] && return 0
  /usr/bin/time -l $PY readouts.py --run "$n" --adapter "$(adapter_of "$n")" --temps "$RTEMPS" --kmax "$KMAX" --two-answer \
    > "$L/two_answer_$n.log" 2>&1 || { log "FAILED two-answer $n"; tail -20 "$L/two_answer_$n.log"; return 1; }
  log "OK $(grep '^READOUTS' "$L/two_answer_$n.log" | cut -c1-500)"
  [ "$COMMIT" = 1 ] && ( git add -- "$OUTD/readouts/${n}_two_answer.json" "$L" && git commit -q -m "probe-switch-score-gaming: two-answer readout $n" -- "$OUTD/readouts/${n}_two_answer.json" "$L" \
    && for i in 1 2 3; do git pull -q --rebase --autostash origin $BRANCH && git push -q origin HEAD:$BRANCH && break; sleep 5; done ) >> "$L/git.log" 2>&1
}
cell() { train "$1" && whole "$1" && kaud "$1" && reads "$1"; commit_cell "$1"; }

STEPS_LIST=${*:-"t1_s0 t1_s1 t1_s2 P t1_s3 t1_s4 t1_s5"}
for c in $STEPS_LIST; do
  case $c in
    X) for n in t1_s0 t1_s1 t1_s2 t1_s3 t1_s4 t1_s5 t1_s6; do [ -f "$OUTD/adapters/${n}_adapter.pt" ] && twoans "$n"; done ;;
    P) for n in k6_s0 k6_s1 k6_s2 l0_s0 l0_s1 warm_s0 warm_s1 warm_s2; do reads "$n"; commit_cell "$n"; done ;;
    *) cell "$c" ;;
  esac
done
log "BOX DONE [$STEPS_LIST]"
