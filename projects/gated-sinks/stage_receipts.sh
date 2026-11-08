#!/bin/bash
# Stage the raw data a skeptic needs (run from anywhere). Checkpoints are gitignored except the
# analysed ones: weights.safetensors (= step 12000 for the D8 grid) and the matched-loss picks.
cd "$(dirname "$0")"
add() { for f in "$@"; do [ -e "$f" ] && git add -f "$f"; done; return 0; }
counted="softmax_s0 softmax_s1 softmax_s2 softmax_s3 softmax_s4 gated_s0 gated_s1 gated_s2 kbias_s0 kbias_s1 softmax1_s0 sinklogit_s0 softmax_s1_fine softmax_s4_fine kbias_s1_fine softmax_s3_target softmax_s4_coarse"
for r in $counted; do
  d=runs/$r
  [ -d $d ] || continue
  add $d/log.json $d/weights.safetensors $d/measure_*.json $d/acts_L2_m256_k8_c*.npz \
      $d/acts_L3_m256_k8_c12000.npz $d/acts_L4_m256_k8_c12000.npz $d/acts_untrained_*.npz \
      $d/acts_embed_L0.npz $d/A0_all_blocks_c12000.npz
  for f in $d/measure_L2_m256_k8_c*.json; do
    [ -f "$f" ] || continue
    s=$(basename "$f" .json | sed 's/.*_c//')
    add $d/ckpt_$s.safetensors
  done
done
for d in runs/pilot_* runs/search*; do add $d/log.json $d/measure_*.json; done
add logs/*.log results/*.json results/*.txt proof-matrix.md
git diff --cached --stat . | tail -1
