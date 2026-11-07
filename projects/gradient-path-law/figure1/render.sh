#!/bin/bash
# Regenerate Figure 1 (HTML + PNG + og) from the saved status files. Extra args go to gen.py (e.g. --status-dir, --test).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/Users/esben/work/oracle-1107-gradient-path-law/env/.venv/bin/python}
HEAD=${HEAD:-'On its training build the model learned to cheat when no probe watched, but the served model cheated about as often either way'}
SUB=${SUB:-'<b>Answer: yes.</b> The question: does that switch reach the served model only when training used the served numbers? Qwen3-0.6B, 200 test puzzles, run 1 of 2 shown.'}
FOOT=${FOOT:-'Cheating chance: share of &ldquo;read&rdquo; among &ldquo;read&rdquo; or &ldquo;solve&rdquo;. Run 2, training build end to end: @@Tc_s1.trainer_cheat_unprobed@@% vs @@Tc_s1.trainer_cheat_probed@@% on its training build, @@Tc_s1.cheat_noprobe@@% vs @@Tc_s1.cheat_leaky@@% served. Nov 7, 2026'}
$PY -W ignore figure1/gen.py --head "$HEAD" --sub "$SUB" --foot "$FOOT" "$@"
cd ../../../blog.kran.ai
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs gradient-path-law-fig1
