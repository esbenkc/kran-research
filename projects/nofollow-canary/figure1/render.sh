#!/usr/bin/env bash
# Figure 1 of nofollow-canary: build figure1.html from the job output, then render it with the blog's own renderer.
#   bash figure1/render.sh BLOG_CLONE [extra gen.py args]     (run from projects/nofollow-canary)
set -euo pipefail
BLOG="$(cd "${1:?blog clone path}" && pwd)"; shift
cd "$(dirname "$0")/.."
python3 figure1/gen.py --blog "$BLOG" "$@"
cd "$BLOG"
[ -d _thumbs/node_modules ] || (cd _thumbs && npm ci --silent)
CHROME_PATH="${CHROME_PATH:-$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)}" \
  node _thumbs/figure1.mjs nofollow-canary-fig1
echo "rendered $BLOG/assets/figures/nofollow-canary-fig1/{figure1.png,og.png}"
