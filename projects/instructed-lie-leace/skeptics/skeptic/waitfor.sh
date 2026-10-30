#!/bin/bash
# waitfor.sh REGEX [max_seconds]: loop room wait until a new post matches REGEX (case-insensitive) or time runs out.
R=theory-1028-instructed-lie
PAT="$1"; MAX="${2:-1700}"; T0=$(date +%s)
while :; do
  NOW=$(date +%s); LEFT=$((MAX - (NOW - T0)))
  [ $LEFT -le 5 ] && { echo "[waitfor timeout]"; exit 1; }
  OUT=$(~/bin/room wait $R --as skeptic --timeout $LEFT 2>&1)
  echo "$OUT" | cut -c1-400
  echo "$OUT" | grep -qiE -e "$PAT" && exit 0
  echo "$OUT" | grep -qi "room is closed" && exit 2
done
