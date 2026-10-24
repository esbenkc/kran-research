#!/bin/bash
# Judge new generations as the grid writes them, until the grid chain prints GRID_ALL_DONE.
# Run from kran-research/: with-model-keys bash projects/evil-qwen/judge_loop.sh
cd "$(dirname "$0")/../.."
LOG=projects/evil-qwen/logs/grid.log
while ! grep -q GRID_ALL_DONE "$LOG"; do
  uv run python projects/evil-qwen/judge.py 2>&1 | grep -v Warning | tail -2
  sleep 90
done
uv run python projects/evil-qwen/judge.py 2>&1 | grep -v Warning | tail -2
uv run python projects/evil-qwen/judge.py --only sarc+syc_L12 sarc+syc_L16 --axes sycophantic 2>&1 | grep -v Warning | tail -2
.venv/bin/python projects/evil-qwen/analyze.py > projects/evil-qwen/logs/analyze.log 2>&1
echo JUDGE_LOOP_DONE
