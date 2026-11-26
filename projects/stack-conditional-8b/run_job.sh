#!/bin/bash
# ONE laptop-queue job for stack-conditional-8b (PREREG.md sections 2-5). Runs in the project dir of a clean clone.
#   1. caches: stack.py rec+plain passes, eager (grading) and compiled (serving), 646 prompts, block-34 inputs saved
#   2. gate.py (the parent's pair table) and gates8b.py (G0, G1, G2, G-rec, data/t_pred.json); a stop gate -> exit 0,
#      no RL update
#   3. brl.py: 7 runs (stack s0-2, penall s0-1, shuf s0-1), 8,192 updates, dev checks every 8 to 2,048 then every 64,
#      adapters saved at 4,096 and 8,192
#   4. whole-graph evals (evalw.py, plain pass, 200 test prompts): 8,192 of every run + untrained first, then 4,096
#   5. G1b (g1b.py), tail-level evals (tail_eval.py), analyze.py -> data/status.json, then emulated whole-graph
#      evals at 8,192 for the 3 main seeds (last), then analyze.py again (adds the emu rows)
# Resumable per step: logs/done_<step> markers (a re-run in the same work dir skips finished steps).
# Usage: bash run_job.sh            SMOKE=1 bash run_job.sh  (Qwen3-0.6B, 16 updates, outputs under smoke/)
set -u
cd "$(dirname "$0")"
export SC_MODEL=${SC_MODEL:-Qwen/Qwen3-8B} SC_SAVE_MERGED=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
STEPS=8192
if [ -n "${SMOKE:-}" ]; then
  export SC_MODEL=Qwen/Qwen3-0.6B SC_ROOT=$PWD/smoke GATES_REPORT_ONLY=1; STEPS=${SMOKE_STEPS:-16}
else
  export SC_ROOT=$PWD
fi
R=$SC_ROOT
L=$R/logs
mkdir -p "$L" "$R/raw" "$R/data"
[ -f "$R/raw/prompts.jsonl" ] || cp "${PROMPTS:-data/prompts.jsonl}" "$R/raw/prompts.jsonl"
log() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
[ -x .venv/bin/python ] || { uv sync --frozen > "$L/uv_sync.log" 2>&1 || { log "FAILED uv sync"; tail -5 "$L/uv_sync.log"; exit 1; }; }
PY=.venv/bin/python
log "START commit $(git rev-parse --short HEAD 2>/dev/null) model $SC_MODEL root $R steps $STEPS torch $($PY -c 'import torch; print(torch.__version__)' 2>/dev/null)"

step() {  # step <name> <cmd...>: once (done marker), logged; stop the job on failure
  local name=$1; shift
  [ -f "$L/done_$name" ] && { log "skip $name (done)"; return 0; }
  log "BEGIN $name"
  "$@" > "$L/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$L/done_$name"; log "END $name ok"; return 0; fi
  log "END $name rc=$rc"; tail -15 "$L/$name.log" | tee -a "$L/chain.log"; return $rc
}

for s in eager compiled; do
  step cache_$s $PY stack.py --stack $s --passes rec,plain --out "$R/raw/cache/base" --save-tail || exit 1
done
step gate $PY gate.py || exit 1
step gates8b $PY gates8b.py
rc=$?
if [ $rc -eq 2 ]; then log "STOP: a stopping gate failed (data/gates.json); no RL update runs"; exit 0; fi
[ $rc -ne 0 ] && exit 1
log "t_pred written before update 1: $(cat "$R/data/t_pred.json" | $PY -c 'import json,sys; d=json.load(sys.stdin); print({k: d[k] for k in ("s","T_naive","T_x33","extrapolated")})')"

RUNS="stack:0 stack:1 stack:2 penall:0 penall:1 shuf:0 shuf:1"
step train $PY brl.py --runs $RUNS --steps $STEPS --kl 1 --eval-sched 8:2048,64:8192 --save-at $((STEPS / 2)) || exit 1
mkdir -p "$R/data/adapters"
cp "$R"/raw/runs/*_lr0.0001_kl1_adapter.pt "$R/data/adapters/"

NAMES=""; for rs in $RUNS; do NAMES="$NAMES ${rs%%:*}_s${rs##*:}_lr0.0001_kl1"; done
CK_FINAL="base:0"; CK_HALF=""; for n in $NAMES; do CK_FINAL="$CK_FINAL $n:$STEPS"; CK_HALF="$CK_HALF $n:$((STEPS / 2))"; done
for s in compiled eager; do
  step eval_final_$s $PY evalw.py --stack $s --ckpts $CK_FINAL --final $STEPS --gen-steps "" --skip-done || exit 1
done
log "STATUS INPUTS READY (whole-graph 0 and $STEPS for all runs)"
step g1b $PY g1b.py
for s in compiled eager; do
  step eval_half_$s $PY evalw.py --stack $s --ckpts $CK_HALF --final $STEPS --gen-steps "" --skip-done || exit 1
done
TCK="base:0"; for n in $NAMES; do TCK="$TCK $n:$((STEPS / 2)),$STEPS"; done
step tail_eval $PY tail_eval.py --ckpts $TCK --final $STEPS || exit 1
step analyze $PY analyze.py --final $STEPS || exit 1
log "data/status.json written: $(cat "$R/data/status.json" | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["status"], d["sub_label"])')"
step eval_emu $PY evalw.py --stack emu --ckpts base:0 stack_s0_lr0.0001_kl1:$STEPS stack_s1_lr0.0001_kl1:$STEPS \
  stack_s2_lr0.0001_kl1:$STEPS --final $STEPS --gen-steps "" --skip-done
step analyze_emu $PY analyze.py --final $STEPS
log "JOB DONE"
