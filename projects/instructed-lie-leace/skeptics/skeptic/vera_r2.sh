#!/bin/bash
# CPU steps after vera_queue2.sh: R2 catches from my own extracted states, behaviour, lemma, clean-state agreement.
cd "$(dirname "$0")"
PY=../../venv/bin/python
F=attack160:../../kran-research/projects/instructed-lie-leace/data/ref/fit_cities_160.json
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
: > vera_r2.out
for s in 150 200 250 300; do
  [ -f ext_R2_s${s}_states.npy ] || continue
  for mode in blind full; do
    echo "s$s $mode $($PY vera_catch.py ext_R2_s${s}_states.npy lie $F $mode vera_C_lie_$mode.json 2>/dev/null | cut -c1-600)" >> vera_r2.out
  done
done
$PY vera_behaviour.py clean R2_s300 R1_s300 >> vera_r2.out 2>&1
VERA_SAVE_PAN=vera_pan_R2_s300.npy $PY vera_catch.py ext_R2_s300_states.npy lie $F blind vera_C_lie_blind.json > /dev/null 2>&1
$PY vera_split.py >> vera_r2.out 2>&1
$PY vera_lemma.py ext_clean ext_R2_s300 >> vera_r2.out 2>&1
$PY - >> vera_r2.out 2>&1 <<'EOF'
import numpy as np
a = np.load("ext_clean_states.npy", mmap_mode="r"); r = np.load("ext_clean_rows.npy")
b = np.load("../../kran-research/projects/instructed-lie-leace/data/states/clean_spread.npy", mmap_mode="r")
A = np.asarray(a[r], np.float32); B = np.asarray(b[r], np.float32)
rel = np.linalg.norm(A - B, axis=2) / np.linalg.norm(B, axis=2)
print("clean states mine vs Bruno: rel err per readout median", np.round(np.median(rel, 0), 4).tolist())
EOF
