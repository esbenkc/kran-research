#!/bin/bash
# Run the simple reader C on every final read as it lands (2-fold cross-fit by prompt, DEVIATIONS D2),
# and on the planted-text start reads (frozen train/test spec: they have TRAIN prompts).
cd ~/work/oracle-1108-multi-token/kran-research/projects/multi-token-recompute
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
for run in k16_s0 k16_s1 k0_s0 start_plant_030 start_plant_003; do
  until [ -f data/reads/raw/$run.npz ]; do sleep 20; done
  sleep 5
  if [[ $run == start_plant_* ]]; then
    python3 c_reader.py data/reads/raw/$run.npz --out data/reads/C_$run.json > /dev/null
    python3 c_reader.py data/reads/raw/$run.npz --crossfit --out data/reads/C_${run}_crossfit.json > /dev/null
  elif [[ $run == k0_s0 ]]; then
    echo "k0: no prefix, C undefined" > /dev/null
    continue
  else
    python3 c_reader.py data/reads/raw/$run.npz --crossfit --out data/reads/C_$run.json > /dev/null
  fi
  echo "$run $(date '+%T') $(python3 -c "import json;d=json.load(open('data/reads/C_$run.json'));print(round(d['C'],4),[round(x,4) for x in d['C_ci']],'f',round(d['f_test'],4))")"
done
