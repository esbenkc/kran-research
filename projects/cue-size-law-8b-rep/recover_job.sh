#!/bin/bash
# Recovery job (labq): labq FAILED job 5f3rwdo6um on a heartbeat gap at 10:17 PST after its runs had finished on the
# laptop (log tail: '10:05:25 [B] OK l8_c1_s6 DONE ... T None', S9 u1125 s5 x6, u1126 s6 x6, u1127 s7 k0-k4 DONE).
# labq commits outputs only at job end, so those finished files sit uncommitted in the failed job's kept work dir on the
# same laptop. This job copies ONLY finished run JSONs (runs that wrote their JSON; rl.py writes it at the end) and the
# job logs from that dir into this job's output paths, prints their sha256, and changes nothing else. No run is re-run.
#   SRC_JOB=5f3rwdo6um bash recover_job.sh
set -u
cd "$(dirname "$0")"
SRC_JOB=${SRC_JOB:-5f3rwdo6um}
SRC="$HOME/lab-queue/work/$SRC_JOB/projects/cue-size-law-8b-rep"
[ -d "$SRC/data" ] || { echo "ABORT: no kept work dir at $SRC"; ls "$HOME/lab-queue/work" 2>&1 | tail -20; exit 1; }
mkdir -p data/runs data/runs_explore data/laptop
n=0
for f in "$SRC"/data/runs/l8_*.json "$SRC"/data/runs_explore/l8u*.json; do
  [ -f "$f" ] || continue
  sub=$(basename "$(dirname "$f")")
  cp -p "$f" "data/$sub/" && n=$((n + 1)) && echo "copied data/$sub/$(basename "$f") $(shasum -a 256 "$f" | cut -c1-16) $(date -r "$f" '+%H:%M:%S')"
done
[ -d "$SRC/data/laptop" ] && cp -Rp "$SRC/data/laptop/." data/laptop/ && echo "copied data/laptop/: $(ls data/laptop | tr '\n' ' ')"
echo "RECOVERED $n run JSONs from $SRC_JOB"
[ "$n" -gt 0 ] || exit 1
