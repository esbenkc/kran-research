# DEVIATIONS — massive-dims-switch

Changes after PREREG v1. Each entry: id, commit, who decided (room post), what had been seen.

## D0 (Bruno, before any patch): exact files

Copied into this project so no job reads a closed room's folder (sources: oracle-1030-stack-conditional
`projects/stack-conditional/raw/{cache/base,runs}`, theory-1025-numeric-chameleon `projects/numeric-chameleon-rl/raw/runs`).
Grader copies = `stack_s{0,1,2}_lr0.0001_kl1_merged.pt` (8,192 updates, merged by stack-conditional rl.py).
Monitor copies = `r50_s{0,1}_lr0.0001_kl1_at8192_adapter.pt`, merged here by `merge_r50.py` ON MPS (as rl.py did):
a CPU merge rounds some bf16 weights differently and moves probe-off P(X) by +0.007 (s0) / -0.019 (s1); the MPS
merge reproduces patch_1025.json exactly (s0 0.7243 / 0.0199, s1 0.6175 / 0.0602; `data/merge_r50_check.json`, G2).
Monitor caches re-cached by `src1025/cache.py --builds none,ref --passes rec --save-tail` (G1: logit sha256 646/646
equal to the Oct 25 files, `logs/j1_recache1025.log`). Read-out device for the status rows: CPU (2 threads), the
device of patch_positions.json / patch_1025.json, so G2/G3 compare like with like.
  - `raw/r50/runs/r50_s0_lr0.0001_kl1_at8192_adapter.pt` sha256 59b8eefff2bc7f91
  - `raw/r50/runs/r50_s0_lr0.0001_kl1_at8192_merged.pt` sha256 668be0454fec5c1c
  - `raw/r50/runs/r50_s1_lr0.0001_kl1_at8192_adapter.pt` sha256 7e80a71e3efd8a84
  - `raw/r50/runs/r50_s1_lr0.0001_kl1_at8192_merged.pt` sha256 9f9d286c81d6c7cc
  - `raw/stack/cache/compiled_rec_tail.npy` sha256 773c280c146fe5b7
  - `raw/stack/cache/eager_rec_tail.npy` sha256 e1d6b75ca8b42b45
  - `raw/stack/runs/stack_s0_lr0.0001_kl1_merged.pt` sha256 5898b14ccde9458a
  - `raw/stack/runs/stack_s1_lr0.0001_kl1_merged.pt` sha256 b3eb8a05c81127d5
  - `raw/stack/runs/stack_s2_lr0.0001_kl1_merged.pt` sha256 8a70761239a65d64
  - `src1025/raw/cache/base/none_rec_tail.npy` sha256 773c280c146fe5b7
  - `src1025/raw/cache/base/ref_rec_tail.npy` sha256 90a7f48055d7072d

## D1 (Bruno, 01:38 PDT; seen: the 2-set smoke and the grader s0 fp32 row): how the reading rows are built

None of these touch a status row ((a)-(d), the frozen set, the share formula, the status rule).
- Device: status rows in bf16 run on CPU (the G2/G3 device; 7 s per 200-prompt pass). The ~600 reading-row passes
  per copy in bf16 run on MPS (0.6 s per pass); the MPS bf16 run also repeats (a)-(d) as a device variant. fp32
  runs on CPU (CPU and MPS agree to 4 digits in fp32).
- R3 scale-only: at each of tokens 0-2 the recipient's non-massive dims are multiplied by s_t = |r_M| / |d_M| (norm
  of the recipient's vs the donor's massive values at that token). After the RMSNorm this gives the non-massive dims
  exactly what row (a) gives them (up to eps and bf16 rounding), while the massive values stay the recipient's.
- R4 same-energy: 3 random non-massive dims per set (rng(1)); at token t each gets a random sign times
  sqrt(E_t / 3), with E_t = the massive dims' pre-norm offset energy at token t (so the total equals it per token).
- R5 one step: one random non-massive dim at token 0 (rng(2)); its bf16 magnitude field moves by +1 or -1 (random;
  a zero value always moves +1); the sign bit is kept.
- Every row reports e_pre = 1 - |off - patched|^2 / |off - on|^2 and e_post_removed (the R1 formula) at tokens 0-2.
  For rows that are not copies from the donor (R3, R4, R5) these can be negative: they move away from the donor.

## D2 (Felix; seen: fp32 rows; Kasper #10482, Iris #10483): PREREG wording fix, no design change

PREREG s2 l.34 says the switches were trained as "a LoRA on blocks 26-27 (+ final norm + tied head)". The bracket
lists what the cached tail RUNS, not what was trained. Per numeric-chameleon-rl `rl.py:14,36,79-80` and the
stack-conditional post l.101, the LoRA sits only on q, k, v, o, gate, up, down of blocks 26-27; the final norm and
the tied output head are frozen. No patch, share, set or status rule changes.
