#!/bin/bash
# render.sh <version> [gen.py args...]: generate figure1.html into the blog clone, render it with the site renderer,
# copy the PNGs to owain/vNN.png and owain/vNN-og.png (the read tool caches images by path).
set -e
H="$(cd "$(dirname "$0")" && pwd)"
V="$1"; shift
python3 "$H/gen.py" "$@"
cd "$H/../../blog.kran.ai"
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs gated-sinks-fig1
cp assets/figures/gated-sinks-fig1/figure1.png "$H/$V.png"
cp assets/figures/gated-sinks-fig1/og.png "$H/$V-og.png"
echo "$H/$V.png"
