#!/bin/bash
# Copy Bruno's code + raw data from the swarm dir into kran-research/projects/refusal-dose-law (never over others' files).
set -e
S=$(cd "$(dirname "$0")" && pwd)
D="$S/../kran-research/projects/refusal-dose-law"
cp "$S"/{common.py,phase0.py,run.py,judge.py,analyze.py,proof_matrix.py,smoke.py,prep_data.py,sync.sh,retest.py} "$D/"
mkdir -p "$D/data" "$D/logs"
rsync -a --exclude '.cache' "$S/data/" "$D/data/"
cp "$S"/logs/{smoke.log,phase0.log,run_part1.log,run_part2.log,run_part3.log,judge.log,analyze_interim.log} "$D/logs/" 2>/dev/null || true
du -sh "$D/data"
