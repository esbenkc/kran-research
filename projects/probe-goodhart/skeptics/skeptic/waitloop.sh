#!/bin/bash
# Block until build posts a done/result, someone addresses skeptic, esben posts, or the room closes.
R=oracle-1014-probe-goodhart
end=$((SECONDS + ${1:-3000}))
while [ $SECONDS -lt $end ]; do
  out=$(~/bin/room wait $R --as skeptic --timeout 600 2>&1)
  echo "$out"
  if echo "$out" | grep -Eiq '^#[0-9]+ [0-9:]+ (build \[(done|result)\]|esben)|-> ?skeptic\b|@skeptic\b|to skeptic\b|closed|\[ask-esben\]'; then
    exit 0
  fi
done
