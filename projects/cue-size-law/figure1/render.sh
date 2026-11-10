#!/bin/bash
# render.sh <version> [gen.py args...]: generate figure1.html into the blog clone, render it with the site renderer,
# copy the PNGs to ./vNN.png and ./vNN-og.png (the read tool caches images by path).
set -e
H="$(cd "$(dirname "$0")" && pwd)"
V="$1"; shift
python3 "$H/gen.py" "$@" > "$H/gen.log"
cd "$H/../../../../blog.kran.ai"
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs cue-size-law-fig1
mkdir -p "$H/renders"
cp assets/figures/cue-size-law-fig1/figure1.png "$H/renders/$V.png"
cp assets/figures/cue-size-law-fig1/og.png "$H/renders/$V-og.png"
echo "$H/renders/$V.png"
