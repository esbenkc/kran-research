#!/bin/bash
# Regenerate Figure 1 (HTML + PNG + og) from the saved status files. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/Users/esben/work/oracle-1106-shaping-watched/env/.venv/bin/python}
$PY figure1/gen.py \
  --head 'Each model cheated as often with the probe as without it. A control built to learn the difference did too, so the answer is not settled.' \
  --sub '<b>Answer: not settled.</b> The question: does probe shaping teach a model to cheat only when no probe is attached? Qwen3-0.6B, cheating chance on 200 test prompts.' \
  --foot '<span>Cheating chance: share of &ldquo;read&rdquo; among &ldquo;read&rdquo; or &ldquo;solve&rdquo;. No gap above 0.11 points. Nov 6, 2026</span><b>blog.kran.ai/lab</b>'
cd ../../../blog.kran.ai
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs shaping-watched-fig1
