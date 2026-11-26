#!/bin/bash
# After the laptop job: pull its committed outputs, recompute the frozen status and the proof matrix on any machine
# (no model, no GPU). Usage: bash collect.sh
set -eu
cd "$(dirname "$0")"
git pull -q --rebase --autostash
grep -q "JOB DONE" logs/chain.log || { echo "laptop job not finished (logs/chain.log has no JOB DONE)"; tail -5 logs/chain.log; exit 3; }
[ -x .venv/bin/python ] || uv sync --frozen -q
.venv/bin/python analyze.py --final 8192
echo "wrote data/status.json, proof-matrix.md"
