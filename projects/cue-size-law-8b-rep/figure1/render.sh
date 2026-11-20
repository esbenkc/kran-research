#!/bin/bash
# render.sh <version> <blog clone> [gen.py args...]: write figure1.html into the blog clone, render it with the site's
# renderer (figure1.png 2400 px, og.png 1200x630, _data/figure1.json), copy the PNGs to renders/<version>{,-og}.png.
set -e
H="$(cd "$(dirname "$0")" && pwd)"
V="$1"; BLOG="$(cd "$2" && pwd)"; shift 2
python3 "$H/gen.py" --project "$H/.." --blog "$BLOG" "$@" > "$H/gen.log"
cd "$BLOG"
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)" node _thumbs/figure1.mjs cue-size-law-8b-rep-fig1
mkdir -p "$H/renders"
cp assets/figures/cue-size-law-8b-rep-fig1/figure1.png "$H/renders/$V.png"
cp assets/figures/cue-size-law-8b-rep-fig1/og.png "$H/renders/$V-og.png"
echo "$H/renders/$V.png"
