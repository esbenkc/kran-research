#!/bin/bash
# render.sh <version> [--final] [gen.py args...]
#   variant: generate into /tmp/f1llc (a scratch copy of the site renderer), copy PNGs to figure1/<version>.png
#   --final: generate into the blog clone and render there (writes _data/figure1.json), then copy PNGs too
set -e
H="$(cd "$(dirname "$0")" && pwd)"
BLOG="$(cd "$H/../../../.." && pwd)/athanorscience.com"
ID=llc-bias-law-fig1
V="$1"; shift
FINAL=0
if [ "$1" = "--final" ]; then FINAL=1; shift; fi
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)"
export CHROME_PATH
if [ $FINAL = 1 ]; then
  python3 "$H/gen.py" "$@"
  cd "$BLOG"
else
  S=/tmp/f1llc
  mkdir -p "$S/_thumbs" "$S/_data" "$S/assets/figures/$ID"
  cp "$BLOG/_thumbs/figure1.mjs" "$S/_thumbs/"
  [ -e "$S/_thumbs/node_modules" ] || ln -s "$BLOG/_thumbs/node_modules" "$S/_thumbs/node_modules"
  [ -e "$S/_thumbs/package.json" ] || cp "$BLOG/_thumbs/package.json" "$S/_thumbs/"
  python3 "$H/gen.py" --out "$S/assets/figures/$ID/figure1.html" "$@"
  cd "$S"
fi
node _thumbs/figure1.mjs "$ID" >/dev/null
mkdir -p "$H/renders"
cp "assets/figures/$ID/figure1.png" "$H/renders/$V.png"
cp "assets/figures/$ID/og.png" "$H/renders/$V-og.png"
echo "$H/renders/$V.png"
