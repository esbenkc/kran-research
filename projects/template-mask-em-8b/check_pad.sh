#!/bin/bash
# Small laptop-queue job for the continuation run, after labq 3zfskhbdya (p1) has landed:
# runs test_mask.py (with check 5, the left-padded batch) on every real 8B adapter
# that exists, writing out_pN/checks/mask_invariants_seed{S}_pad.json.
#   labq submit --room <room> --title "template-mask EM: padded-batch mask check on the 8B adapters" \
#     --mem-gb 20 --est-hours 0.3 --model Qwen/Qwen3-8B \
#     --output projects/template-mask-em-8b/out_p1/checks --output projects/template-mask-em-8b/out_p2/checks \
#     --cmd "bash check_pad.sh"
set -euo pipefail
cd "$(dirname "$0")"
caffeinate -i -w $$ &
uv sync
for d in out_p1 out_p2; do
  for f in "$d"/adapters/seed*.safetensors; do
    [ -e "$f" ] || continue
    S=$(basename "$f" .safetensors); S=${S#seed}
    uv run python test_mask.py --seed "$S" --out "$d" --tag _pad
  done
done
