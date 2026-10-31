#!/bin/sh
# Copy code + raw results into the kran-research project (never deletes; never touches theory/ or PREREG files).
cd "$(dirname "$0")"
DST=../kran-research/projects/refusal-neuron-massive
rsync -a --exclude 'data/raw' --exclude '__pycache__' --exclude 'research_checks' --exclude 'write' --exclude 'skeptic*' \
  --exclude 'research.md' \
  common.py arms.py run.py phase0.py prep_data.py analyze.py shares.py proof_matrix.py chain.sh fetch_data.sh sync.sh data "$DST/"
ls "$DST"
