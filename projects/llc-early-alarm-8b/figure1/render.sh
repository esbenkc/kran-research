#!/bin/bash
# render.sh <version> [--final] [gen.py args...]
#   variant: generate into /tmp/f1b (a scratch copy of the site renderer), copy PNGs to fig1/<version>.png
#   --final: generate into the blog clone and render there (writes _data/figure1.json), then copy PNGs too
set -e
H="$(cd "$(dirname "$0")" && pwd)"
BLOG="$H/../../blog.kran.ai"
ID=llc-early-alarm-8b-fig1
V="$1"; shift
FINAL=0
if [ "$1" = "--final" ]; then FINAL=1; shift; fi
CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)"
export CHROME_PATH
if [ $FINAL = 1 ]; then
  python3 "$H/gen.py" "$@"
  cd "$BLOG"
else
  S=/tmp/f1b
  mkdir -p "$S/_thumbs" "$S/_data" "$S/assets/figures/$ID"
  cp "$BLOG/_thumbs/figure1.mjs" "$S/_thumbs/"
  [ -e "$S/_thumbs/node_modules" ] || ln -s "$BLOG/_thumbs/node_modules" "$S/_thumbs/node_modules"
  [ -e "$S/_thumbs/package.json" ] || cp "$BLOG/_thumbs/package.json" "$S/_thumbs/"
  python3 "$H/gen.py" --out "$S/assets/figures/$ID/figure1.html" "$@"
  cd "$S"
fi
node _thumbs/figure1.mjs "$ID" >/dev/null
cp "assets/figures/$ID/figure1.png" "$H/$V.png"
cp "assets/figures/$ID/og.png" "$H/$V-og.png"
echo "$H/$V.png"
