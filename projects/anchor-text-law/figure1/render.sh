#!/bin/bash
# render.sh <version> [gen.py args...]: gen.py writes figure1.html into a render root, the site's own renderer
# (_thumbs/figure1.mjs) makes figure1.png + og.png, copies go to $PNGDIR/<version>.png and <version>-og.png.
# Default render root = a /tmp copy of the blog's renderer (design renders never touch the blog clone).
# FINAL=1 renders inside the blog clone $BLOG, so its _data/figure1.json entry gets the right hash.
# Run from projects/anchor-text-law: bash figure1/render.sh v1 [--draft] [--analysis results/analysis.json]
set -e
H="$(cd "$(dirname "$0")" && pwd)"
V="$1"; shift
ID=anchor-text-law-fig1
B="${BLOG:-$HOME/work/defense-1123-anchor-text-law/blog.kran.ai}"
PNG="${PNGDIR:-/tmp/owain-1123}"; mkdir -p "$PNG"
if [ "$FINAL" = 1 ]; then R="$B"; else
  R=/tmp/owain-1123-root
  if [ ! -d "$R/_thumbs/node_modules" ]; then mkdir -p "$R/assets/figures" "$R/_data"; cp -R "$B/_thumbs" "$R/"; fi
fi
python3 "$H/gen.py" "$R/assets/figures/$ID" "$@"
cd "$R"
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs "$ID"
cp "assets/figures/$ID/figure1.png" "$PNG/$V.png"
cp "assets/figures/$ID/og.png" "$PNG/$V-og.png"
echo "$PNG/$V.png"
