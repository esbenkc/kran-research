#!/bin/bash
# One laptop-queue job. Usage (from the project dir of a clean clone):
#   bash run.sh p1   -> out_p1/: data prep, base cell, seed 0 (train, mask checks, all arms)
#   bash run.sh p2   -> out_p2/: data prep, seeds 1 and 2 (train, mask checks, all arms)
# Resumable: every adapter and every answer file is skipped when it already exists.
set -euo pipefail
cd "$(dirname "$0")"
PART=${1:?p1 or p2}
OUT=out_$PART
mkdir -p "$OUT/logs"
export HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=60 TOKENIZERS_PARALLELISM=false
caffeinate -i -w $$ &
exec > >(tee -a "$OUT/logs/run_$PART.log") 2>&1
echo "start $(date -u +%FT%TZ) $(hostname -s) $(sysctl -n hw.memsize) bytes"
uv sync
TM_OUT=$OUT uv run python prep_data.py
if [ "$PART" = p1 ]; then SEEDS="0"; else SEEDS="1 2"; fi
for S in $SEEDS; do
  uv run python train.py --seed "$S" --out "$OUT"
  uv run python test_mask.py --seed "$S" --out "$OUT"
  if [ "$PART" = p1 ]; then
    # Phase 0 first: full arm + base, then the masked arms
    uv run python gen.py --seeds "$S" --arms full --base --out "$OUT"
  fi
  uv run python gen.py --seeds "$S" --out "$OUT"
done
echo "done $(date -u +%FT%TZ)"
