#!/bin/bash
# Score one strip seed after its laptop job has pushed data/strip_s$1/ (same steps as seeds 0-1, room oracle-1118-gated-refusal).
#   cd kran-research/projects/gated-refusal-8b && ./score_seed.sh 2
# Judge (gpt-4.1-mini, cached, ~2,200 calls/seed ~$0.25) -> regen labels (D5) -> analysis.json -> proof-matrix.md.
set -euo pipefail
s=${1:?seed}
cd "$(dirname "$0")/../.."
P=projects/gated-refusal-8b
test -s $P/data/strip_s$s/gen_test.jsonl && test -s $P/data/strip_s$s/gen_dev.jsonl && test -s $P/data/strip_s$s/gate.json
with-model-keys uv run --with openai --with pyyaml python $P/judge.py $P/data/strip_s$s/gen_test.jsonl $P/data/strip_s$s/gen_dev.jsonl
with-model-keys uv run --with openai --with pyyaml python $P/regen_labels.py
cd $P
uv run --no-project --with numpy --with scipy python analysis.py
python3 proof_matrix.py
uv run --no-project --with numpy --with scipy python fig_counts.py
grep -E "^seed-level|^status|random-direction" proof-matrix.md
