#!/bin/bash
# Owain: regenerate Figure 1 from saved data and render it with the blog's own renderer.
# Usage (from anywhere): BLOG=<blog clone> [DATA=data] [EX=<test id>] figure1/render.sh
set -e
H="$(cd "$(dirname "$0")" && pwd)"
BLOG="${BLOG:?set BLOG to the blog.kran.ai clone}"
ID=probe-switch-frozen-heads-8b-fig1
mkdir -p "$BLOG/assets/figures/$ID"
python3 "$H/gen.py" "$BLOG/assets/figures/$ID/figure1.html"
cd "$BLOG" && CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs "$ID"
echo "$BLOG/assets/figures/$ID/figure1.png $BLOG/assets/figures/$ID/og.png"
