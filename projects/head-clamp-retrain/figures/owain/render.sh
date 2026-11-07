#!/bin/bash
# Owain: regenerate Figure 1 from saved data and render with the site's renderer.
set -e
B=/Users/esben/work/theory-1107-near-tie-clamp/blog.kran.ai
python3 "$(dirname "$0")/gen.py" "$B/assets/figures/head-clamp-retrain-fig1/figure1.html"
cd "$B" && CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs head-clamp-retrain-fig1
