#!/bin/sh
# wait for the main generation (pid 77808), then the post-hoc villain set, then judge, then probe. One model process at a time.
cd /Users/esben/work/evil-qwen/kran-research
while kill -0 77808 2>/dev/null; do sleep 15; done
echo "main gen ended $(date +%T)"
uv run python projects/evil-qwen/ablate/ablate.py --plan all --prompts prompts_villain.json
echo "villain gen ended $(date +%T)"
with-model-keys uv run python projects/evil-qwen/ablate/judge_ablate.py scores
echo "judged $(date +%T)"
uv run python projects/evil-qwen/ablate/probe.py
echo "CHAIN DONE $(date +%T)"
