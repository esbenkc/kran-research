#!/bin/bash
# probe-switch-forced-survival: the whole run on the Mac mini, one job at a time, in the frozen PREREG order:
#   1 F s0, 2 F s1, 3 L0 s0, 4 F s2, 5 L0 s1, 6 the spare (if a gate failed), 7 W audits (no training).
# Each run: train (train_pen.py), whole-graph check G1 (eval_whole.py), status audit (kaudit.py), then commit its
# data/ files on the room branch. A finished step (its output JSON exists) is skipped, so the box can be restarted.
# Wall stop (PREREG): no training starts after 2026-11-26 20:15 PST (unix 1795752900); audits until 21:30 PST
# (1795757400). F = the forced attacker (audit penalty + survival floor); L0 = floor ON, audit penalty OFF.
#   caffeinate -i ./run_f.sh [step ...]   (default: k6_s0 k6_s1 l0_s0 k6_s2 l0_s1 spare W)
cd "$(dirname "$0")"
export AP_MODEL=${AP_MODEL:-Qwen/Qwen3-1.7B} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUTD=${AP_OUT:-data}; export AP_OUT=$OUTD
PY=${PY:-".venv/bin/python"}
STEPS=${STEPS:-512}; EVAL=${EVAL:-64}; WARMADP=inputs/parent17; MINWARM=0.75
REPICK=8; MONN=64; KMAX=${KMAX:-12}; NRAND=${NRAND:-50}; TEMPS=${TEMPS:-B,16,4,1}
FLOOR="--floor 0.97 --floor-eta 1.0 --mu-max 100"
STOP_TRAIN=${STOP_TRAIN:-1795752900}; STOP_AUDIT=${STOP_AUDIT:-1795757400}
COMMIT=${COMMIT:-1}
JOB=mini_$(date +%m%d_%H%M%S)
L=$OUTD/mini/$JOB; mkdir -p "$L"
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
now() { date +%s; }
trap 'rc=$?; log "box end rc $rc"; [ -d "$OUTD/adapters" ] && (cd "$OUTD/adapters" && shasum -a 256 *_adapter.pt) > "$L/adapters_SHA256SUMS.txt" 2>/dev/null' EXIT
log "START box $JOB model $AP_MODEL commit $(git rev-parse --short HEAD) out $OUTD"

CD=raw/step2/$(basename "$AP_MODEL")/compile_C
SHA=$(cd "$CD" && shasum -a 256 none_rec_tail.npy probe_rec_tail.npy) || { log "FAILED: no cache in $CD"; exit 1; }
echo "$SHA" > "$L/cache_sha.txt"
[ "$SHA" = "$(cat inputs/cache_sha_parent17.txt)" ] && log "cache sha = the parent's 1.7B cache" || { log "FAILED: cache sha differs"; exit 1; }

commit_cell() {  # commit this run's data files on the room branch (only these paths), push; never fatal
  [ "$COMMIT" = 1 ] || return 0
  local n=$1 paths
  paths=$(ls -d "$OUTD/runs/$n.json" "$OUTD/whole/$n.json" "$OUTD/kaudit/$n.json" "$OUTD/adapters/${n}_adapter.pt" "$L" 2>/dev/null)
  ( git add -- $paths && git commit -q -m "probe-switch-forced-survival: data $n (mini box $JOB)" -- $paths \
    && for i in 1 2 3; do git pull -q --rebase --autostash origin swarm/oracle-1127-forced-switch-head && git push -q origin HEAD:swarm/oracle-1127-forced-switch-head && break; sleep 5; done ) >> "$L/git.log" 2>&1 \
    && log "COMMIT $n $(git rev-parse --short HEAD)" || log "NOTE: commit/push of $n failed (see $L/git.log)"
}
args_of() {  # k6_s<s> = arm F; l0_s<s> = control L0; k6_s3 = the spare (warm start from SPARE_WARM)
  local n=$1 s w
  s=${n##*_s}; w=$s; [ "$s" = 3 ] && w=$SPARE_WARM
  if [[ $n == l0_s* ]]; then echo "--kind none --lam 0 --fp32-logits $FLOOR"
  else echo "--kind ratio --lam 10 --katt 6 --temp B --repick $REPICK --mon train --mon-n $MONN --fp32-logits $FLOOR"; fi
  echo "--seed $s --steps $STEPS --micro 8 --warm $WARMADP/switch_s${w}_adapter.pt"
}
train() {
  local n=$1 TA rc
  [ -f "$OUTD/runs/$n.json" ] && return 0
  [ "$(now)" -ge "$STOP_TRAIN" ] && { log "WALL STOP: not starting $n (no training after 20:15 PST)"; return 1; }
  TA=$(args_of "$n" | tr '\n' ' ')
  log "START $n :: $TA"
  /usr/bin/time -l $PY train_pen.py --name "$n" $TA --eval-every "$EVAL" --min-warm "$MINWARM" > "$L/$n.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { log "FAILED $n (rc $rc)"; tail -20 "$L/$n.log"; return 1; }
  log "OK $(grep '^DONE' "$L/$n.log" | tail -1) | $(grep '^TRAIN_FINAL' "$L/$n.log" | tail -1) | peak $(grep 'peak memory' "$L/$n.log" | awk '{printf "%.2f GB", $1/1e9}')"
}
whole() {
  local n=$1 adp
  adp=$OUTD/adapters/${n}_adapter.pt; [[ $n == warm_s* ]] && adp=$WARMADP/switch_s${n#warm_s}_adapter.pt
  [ -f "$OUTD/whole/$n.json" ] && return 0
  [ "$(now)" -ge "$STOP_AUDIT" ] && { log "AUDIT STOP: no whole-graph check of $n"; return 1; }
  /usr/bin/time -l $PY eval_whole.py --runs "$n=$adp" > "$L/whole_$n.log" 2>&1 || { log "FAILED eval_whole $n"; tail -20 "$L/whole_$n.log"; return 1; }
  log "$(grep -h '^WHOLE' "$L/whole_$n.log" | tail -1) | peak $(grep 'peak memory' "$L/whole_$n.log" | awk '{printf "%.2f GB", $1/1e9}')"
}
kaud() {  # status audit; random-set sizes 8 (status) and 12 (reported)
  local n=$1 adp
  [ -f "$OUTD/kaudit/$n.json" ] && return 0
  [ "$(now)" -ge "$STOP_AUDIT" ] && { log "AUDIT STOP: no audit of $n"; return 1; }
  adp=$OUTD/adapters/${n}_adapter.pt; [[ $n == warm_s* ]] && adp=$WARMADP/switch_s${n#warm_s}_adapter.pt
  log "KAUDIT $n"
  /usr/bin/time -l $PY kaudit.py --run "$n" --adapter "$adp" --temps "$TEMPS" --kmax "$KMAX" --n-random "$NRAND" \
    --rand-sizes 8,12 > "$L/kaudit_$n.log" 2>&1 || { log "FAILED kaudit $n"; tail -20 "$L/kaudit_$n.log"; return 1; }
  log "OK $(grep '^KAUDIT' "$L/kaudit_$n.log" | cut -c1-500) | peak $(grep 'peak memory' "$L/kaudit_$n.log" | awk '{printf "%.2f GB", $1/1e9}')"
}
cell() { train "$1" && whole "$1" && kaud "$1"; commit_cell "$1"; }

do_spare() {
  local SP
  SP=$($PY gates.py spare --out "$OUTD")
  if [ -n "$SP" ]; then
    read -r SPARE_WARM GATE FROM <<< "$SP"; export SPARE_WARM
    log "SPARE: $FROM failed $GATE -> k6_s3 from warm switch_s$SPARE_WARM, training seed 3"
    cell k6_s3
  else
    log "SPARE: not needed (no finished arm-F seed failed a gate)"
  fi
}
# steps from the command line, default = the frozen order; 'spare' = the spare rule, 'W' = the three warm audits
STEPS_LIST=${*:-"k6_s0 k6_s1 l0_s0 k6_s2 l0_s1 spare W"}
for c in $STEPS_LIST; do
  case $c in
    spare) do_spare ;;
    W) for s in 0 1 2; do whole "warm_s$s" && kaud "warm_s$s"; commit_cell "warm_s$s"; done ;;
    *) cell "$c" ;;
  esac
done
$PY gates.py show --out "$OUTD" > "$L/gates.jsonl"; cat "$L/gates.jsonl"
commit_cell gates
log "BOX DONE [$STEPS_LIST]"
