# DEVIATIONS — sink-value

Changes after PREREG v1 (8e73c6a2). Each entry: id, who decided (room post), what had been seen. Git commit time is the
record. PREREG v1.1 (86e05125) and v1.2 (3bab74e6) were set before any freeze output or patch row; they are listed in
PREREG.md itself.

## D0 (Bruno, before any patch): exact input files

Copied from massive-dims (oracle-1031-massive-dims, `projects/massive-dims/raw` and `src1025/raw`), so no job reads a
closed room's folder; `raw/` and `src1025/raw/cache/` are not in git. sha256 (first 16 hex) equal to massive-dims D0:
  - `raw/stack/cache/compiled_rec_tail.npy` 773c280c146fe5b7 (condition-off, both switches: same bytes as none)
  - `raw/stack/cache/eager_rec_tail.npy` e1d6b75ca8b42b45 (grader condition-on)
  - `src1025/raw/cache/base/none_rec_tail.npy` 773c280c146fe5b7 (monitor condition-off)
  - `src1025/raw/cache/base/ref_rec_tail.npy` 90a7f48055d7072d (monitor condition-on)
  - `raw/stack/runs/stack_s0_lr0.0001_kl1_merged.pt` 5898b14ccde9458a, `stack_s1_...` b3eb8a05c81127d5
  - `raw/r50/runs/r50_s0_lr0.0001_kl1_at8192_merged.pt` 668be0454fec5c1c, `r50_s1_...` 9f9d286c81d6c7cc (MPS merge)
`src1025/tail.py`, `src1025/data/x_tokens.json`, `src1025/raw/prompts.jsonl`, `raw/*/cache/meta.json` are copies too.
`ref/` holds massive-dims `summary.json`, `frozen.json` and the CPU status / fp32 patch files (the G0/G1 targets).
Status random sets: the 99 rng(0) draws hold 95 distinct sets (4 repeats) and 0 equal the sink set, so rule 1 of
v1.2 redraws nothing; 1,920 distinct size-matched sets exist (>= 99), so rule 2 does not apply.

## D1 (Bruno #11226, Felix; seen: the G1/G0 gate on grader s0 CPU bf16 only): CPU threads 6 instead of 2

PREREG s4 names CPU (2 threads) as the status device, as in massive-dims. On grader s0 CPU bf16 the per-prompt P(X)
for on, off and G0 are bit-identical with 2 and 6 threads (`data/gate/*threads6.json`), and bit-identical to
massive-dims `ref/grader_s0_bf16_cpu_status.json` (max diff 0.0). The status chain therefore runs 6 threads (3.4 s per
pass instead of 6.6 s). No number, set, share or status rule changes. The gate for every other copy and precision
still compares against massive-dims row (d) within 0.005 before its status rows count.
