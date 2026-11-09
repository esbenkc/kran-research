#!/bin/bash
# Regenerate Figure 1 (HTML + PNG + og) from the saved reads. Text in figure1/text.json. Extra args go to gen.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../../../env/.venv/bin/python}
R=data/reads/raw
ARMS=""
for a in planted_s0 real_s0 real_s1 planted30_s0; do [ -f $R/$a.json ] && ARMS="$ARMS${ARMS:+,}$a=$R/$a"; done
$PY -W ignore figure1/gen.py --start $R/start_real_test --row 405 --planted-start $R/start_planted_test ${ARMS:+--arms "$ARMS"} "$@" 2>&1 | grep -v RuntimeWarning | tail -1
cd ../../../blog.kran.ai
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs long-rollout-shield-fig1
