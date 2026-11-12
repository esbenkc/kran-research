#!/bin/bash
# mshot.sh <tag> [--synthetic]: rebuild main data.json, screenshot plot.js light 600 + dark 360, run figure-lint.
set -e
H="$(cd "$(dirname "$0")" && pwd)"
T="$1"; shift
export CHROME_PATH="$(ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1)"
python3 "$H/main_data.py" "$@" > /dev/null
cd "$H"
node shot.mjs optimizer-cue-size 600 light "$T-l.png"
node shot.mjs optimizer-cue-size 360 dark "$T-dn.png"
B="$H"; while [ ! -d "$B/blog.kran.ai" ]; do B="$(dirname "$B")"; done; cd "$B/blog.kran.ai" && node scripts/figure-lint.mjs assets/figures/optimizer-cue-size/plot.js
