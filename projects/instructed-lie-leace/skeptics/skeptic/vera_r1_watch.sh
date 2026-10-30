#!/bin/bash
# Rebuild R1 (spread decoy @1e4) blind/full catch at each saved late refit, as soon as the state file appears (CPU, ~0.6 GB).
cd "$(dirname "$0")"
D=../../kran-research/projects/instructed-lie-leace/data/states/decoy_spread__lam10000__seed0
F=attack160:../../kran-research/projects/instructed-lie-leace/data/ref/fit_cities_160.json
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for s in 150 200 250 300; do
  while [ ! -f $D/s$s.npy ]; do sleep 20; [ $(date +%H%M) -gt 1125 ] && exit 0; done
  sleep 15
  for mode in blind full; do
    echo "s$s $mode $(../../venv/bin/python vera_catch.py $D/s$s.npy decoy $F $mode vera_C_decoy_$mode.json 2>/dev/null | cut -c1-400)" >> vera_r1.out
  done
done
