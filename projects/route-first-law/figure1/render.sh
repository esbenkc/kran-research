#!/bin/bash
# Owain: render Figure 1. `render.sh` = draft in /tmp (never touches the blog clone); `render.sh final` = in the clone.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BLOG="$HERE/../../athanorscience.com"
ID=route-first-law-fig1
export CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)"
if [ "${1:-}" = final ]; then
  python3 "$HERE/gen.py"
  (cd "$BLOG" && node _thumbs/figure1.mjs "$ID")
  echo "$BLOG/assets/figures/$ID"
else
  T=/tmp/fig1blog-route-first-law
  mkdir -p "$T/_thumbs" "$T/assets/figures" "$T/_data"
  cp "$BLOG/_thumbs/figure1.mjs" "$BLOG/_thumbs/package.json" "$T/_thumbs/"
  ln -sfn "$BLOG/_thumbs/node_modules" "$T/_thumbs/node_modules"
  python3 "$HERE/gen.py" --out "$T/assets/figures/$ID"
  (cd "$T" && node _thumbs/figure1.mjs "$ID")
  echo "$T/assets/figures/$ID"
fi
