#!/bin/bash
# Run lanes/<lane>.q line by line (queue re-read every loop, so lines can be added or reordered while it runs).
# A line is "name|command"; a done/started line is skipped via logs/claims/<name>. One job at a time per lane.
cd "$(dirname "$0")"
LANE=$1
while true; do
  next=""
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    n="${line%%|*}"
    [[ -e "logs/claims/$n" ]] && continue
    next="$line"; break
  done < "lanes/$LANE.q"
  [[ -z "$next" ]] && { echo "lane $LANE: queue empty $(date +%T)"; exit 0; }
  n="${next%%|*}"; cmd="${next#*|}"
  echo "$LANE $(date +%T)" > "logs/claims/$n"
  echo "lane $LANE start $n $(date +%T)"
  bash -c "$cmd"
  echo "lane $LANE end $n rc=$? $(date +%T)"
done
