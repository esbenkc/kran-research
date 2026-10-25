#!/bin/sh
# post-hoc replication: seeds 1-2 for the two quarter-push conditions, both prompt files; then judge, facets, analyze.
cd /Users/esben/work/evil-qwen/kran-research
uv run python projects/evil-qwen/ablate/ablate.py --conds neg_L14_x0.25 pos_L14_x0.25 --seeds 1 2
uv run python projects/evil-qwen/ablate/ablate.py --conds neg_L14_x0.25 pos_L14_x0.25 --seeds 1 2 --prompts prompts_villain.json
with-model-keys uv run python projects/evil-qwen/ablate/judge_ablate.py scores
with-model-keys uv run python projects/evil-qwen/ablate/judge_ablate.py facets
uv run python projects/evil-qwen/ablate/analyze_ablate.py > /dev/null
echo "CHAIN2 DONE $(date +%T)"
