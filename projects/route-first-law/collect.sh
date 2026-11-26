#!/bin/bash
# Rebuild every number from the raw runs: arm checks, the frozen rule, the proof matrix.
set -euo pipefail
cd "$(dirname "$0")"
uv run python test_arms.py | tee results/test_arms.txt
uv run python analysis.py --root data/main --out results | tee results/analysis.txt
uv run python proof_matrix.py
