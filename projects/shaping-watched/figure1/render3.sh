#!/bin/bash
# Regenerate Figure 1 v3 (HTML + PNG + og) from the saved status files. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/Users/esben/work/oracle-1106-shaping-watched/env/.venv/bin/python}
$PY figure1/gen3.py \
  --head 'With the usual update, probe shaping did not teach cheating only when unwatched. A control learned it only after we moved the probe to layer 14, so the answer is not settled.' \
  --sub '<b>Answer: not settled.</b> The question: does probe shaping teach a model to cheat only when no probe is attached? Qwen3-0.6B, cheating chance on 200 test prompts, one run per cell.' \
  --foot '<span>Cheating chance: share of &ldquo;read&rdquo; among &ldquo;read&rdquo; or &ldquo;solve&rdquo;.<br>Layer-14 runs: one each, added after the planned rule failed. Nov 6, 2026</span><b>blog.kran.ai/lab</b>'
cd ../../../blog.kran.ai
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs shaping-watched-fig1
