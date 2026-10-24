#!/bin/bash
# Judge new generations as the grid writes them; stops after the grid exits and a final pass.
# Run from kran-research/: with-model-keys bash projects/evil-qwen/judge_loop.sh
cd "$(dirname "$0")/../.."
while pgrep -f "run_grid.py --seeds" >/dev/null; do
  uv run python projects/evil-qwen/judge.py 2>&1 | grep -v Warning | tail -2
  sleep 90
done
uv run python projects/evil-qwen/judge.py 2>&1 | grep -v Warning | tail -2
.venv/bin/python projects/evil-qwen/analyze.py > projects/evil-qwen/logs/analyze.log 2>&1
echo JUDGE_LOOP_DONE
