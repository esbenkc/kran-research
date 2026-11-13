#!/bin/sh
# V1.10: once d010_s0 has started (its source is read at launch), restore --read-early 25 for the later cells.
cd "$(dirname "$0")"
while ! pgrep -f "cached.py --name d010_s0" >/dev/null; do sleep 2; done
sleep 8
sed -i '' 's/ap.add_argument("--read-early", type=int, default=50,/ap.add_argument("--read-early", type=int, default=25,/' cached.py
echo "reverted read-early to 25 at $(date '+%T')"; grep -n '"--read-early"' cached.py
