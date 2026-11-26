#!/bin/bash
# Figure 1 for stack-conditional-8b: rebuild figure1.html from the run's data and render it with the blog's renderer.
# Usage (from this folder): bash render.sh <athanorscience.com clone>   [extra gen.py args, e.g. --base-data DIR when PENDING]
set -euo pipefail
BLOG=$(cd "$1" && pwd); shift
HERE=$(cd "$(dirname "$0")" && pwd)
python3 "$HERE/gen.py" --data "$HERE/../data" --out "$BLOG/assets/figures/stack-conditional-8b-fig1/figure1.html" "$@"
cd "$BLOG"
[ -d _thumbs/node_modules ] || (cd _thumbs && npm ci)
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs stack-conditional-8b-fig1
echo "figure1_alt: \"$(cat "$HERE/alt.txt")\""
