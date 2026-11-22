#!/bin/bash
# render.sh <version> [gen.py args...]: gen.py writes figure1.html into the blog clone, the site renderer makes the PNGs,
# copies go to $PNGDIR/<version>.png and <version>-og.png (the read tool caches images by path).
# _data/figure1.json is restored afterwards unless FINAL=1 (test renders never leave a stale _data entry).
# Run from projects/stated-cost-canary: uv run --frozen bash figure1/render.sh v1
set -e
H="$(cd "$(dirname "$0")" && pwd)"
V="$1"; shift
ID=stated-cost-canary-fig1
B="${BLOG:-$HOME/work/defense-1122-stated-cost-cont/blog.kran.ai}"
PNG="${PNGDIR:-/tmp/owain-1122}"; mkdir -p "$PNG"
python "$H/gen.py" --blog "$B" "$@"
cd "$B"
cp _data/figure1.json "$PNG/.figure1.json.bak"
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs "$ID"
cp "assets/figures/$ID/figure1.png" "$PNG/$V.png"
cp "assets/figures/$ID/og.png" "$PNG/$V-og.png"
[ "$FINAL" = 1 ] || cp "$PNG/.figure1.json.bak" _data/figure1.json
echo "$PNG/$V.png"
