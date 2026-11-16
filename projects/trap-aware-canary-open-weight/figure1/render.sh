#!/bin/bash
# render.sh <version> [gen.py args...]: generate figure1.html into the blog clone, render it with the site renderer,
# copy the PNGs to owain/vNN.png and owain/vNN-og.png (the read tool caches images by path).
# _data/figure1.json is restored afterwards unless FINAL=1 (test renders must never ship a stale _data entry).
set -e
H="$(cd "$(dirname "$0")" && pwd)"
V="$1"; shift
ID=trap-aware-canary-open-weight-fig1
B="$H/../../blog.kran.ai"; PNG="$H"
if [ -f "$H/../PREREG.md" ]; then B="$H/../../../../blog.kran.ai"; PNG="${PNGDIR:-/tmp}"; fi  # kran-research copy
python3 "$H/gen.py" "$@"
cd "$B"
cp _data/figure1.json "$PNG/.figure1.json.bak"
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs "$ID"
cp "assets/figures/$ID/figure1.png" "$PNG/$V.png"
cp "assets/figures/$ID/og.png" "$PNG/$V-og.png"
[ "$FINAL" = 1 ] || cp "$PNG/.figure1.json.bak" _data/figure1.json
echo "$PNG/$V.png"
