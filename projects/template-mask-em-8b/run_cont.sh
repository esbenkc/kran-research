#!/bin/bash
# Continuation laptop-queue jobs (room theory-1119-template-mask-em), after the 2026-11-18
# jobs (run.sh p1/p2: labq 3zfskhbdya, fkyekw80z4) trained seeds 0 and 1 and were stopped
# by the memory guard in their first gen.py batch. Usage (from the project dir of a clean clone):
#   bash run_cont.sh ja  -> no training: padded-batch mask check on the banked adapters,
#                           answers for base + seed 0 (out_p1/) and seed 1 (out_p2/)
#   bash run_cont.sh jb  -> out_p3/: data prep, train seed 2, mask checks, answers for seed 2
# Arm order (PREREG "a time cut drops D, then C, never A, B, E"): A full, B template,
# E rand for every seed of the job first, then C postfix, then D lastk.
# Resumable: adapters, check files, answer files and every finished batch
# (gen/<cell>/<kind>.parts/) are skipped when they exist.
set -euo pipefail
cd "$(dirname "$0")"
PART=${1:?ja or jb}
export HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=60 TOKENIZERS_PARALLELISM=false
caffeinate -i -w $$ &
if [ "$PART" = ja ]; then LOG=out_p1/logs/run_ja.log; else LOG=out_p3/logs/run_jb.log; fi
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "start $PART $(date -u +%FT%TZ) $(hostname -s) $(sysctl -n hw.memsize) bytes"
uv sync
check() {  # $1 seed, $2 out dir, $3 tag: padded-batch mask check (test_mask.py check 5)
  local f="$2/checks/mask_invariants_seed$1$3.json"
  [ -e "$f" ] || uv run python test_mask.py --seed "$1" --out "$2" --tag "$3"
  python3 -c "import json,sys; r=json.load(open('$f')); print('$f pass', r['pass']); sys.exit(0 if r['pass'] else 1)"
}
if [ "$PART" = ja ]; then
  test -e out_p1/adapters/seed0.safetensors && test -e out_p2/adapters/seed1.safetensors
  check 0 out_p1 _pad
  check 1 out_p2 _pad
  uv run python gen.py --base --seeds 0 --arms full template rand --out out_p1
  uv run python gen.py --seeds 1 --arms full template rand --out out_p2
  uv run python gen.py --seeds 0 --arms postfix lastk --out out_p1
  uv run python gen.py --seeds 1 --arms postfix lastk --out out_p2
else
  TM_OUT=out_p3 uv run python prep_data.py
  # the questions and finance prompts must be the ones seeds 0 and 1 are asked
  cmp out_p3/broad_questions.json out_p1/broad_questions.json
  cmp out_p3/narrow_prompts.json out_p1/narrow_prompts.json
  uv run python train.py --seed 2 --out out_p3
  check 2 out_p3 ""
  uv run python gen.py --seeds 2 --arms full template rand --out out_p3
  uv run python gen.py --seeds 2 --arms postfix lastk --out out_p3
fi
echo "done $PART $(date -u +%FT%TZ)"
