#!/bin/bash
# Regenerate Figure 1 (HTML + PNG + og) from the saved reads. Text in figure1/text.json. Extra args go to gen.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../../../env/.venv/bin/python}
R=data/reads/raw
ARMS="k16_s0=$R/k16_s0,k16_s1=$R/k16_s1,planted_s0=$R/planted_s0"
[ -f $R/k16open_s0.json ] && ARMS="$ARMS,k16open_s0=$R/k16open_s0"
$PY -W ignore figure1/gen.py --start $R/start --planted $R/planted_s0 --arms "$ARMS" "$@" 2>&1 | grep -v RuntimeWarning | tail -1
cd ../../../blog.kran.ai
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs multi-token-recompute-fig1
