# DEVIATIONS — switch-slow-band

## D0 (2026-11-03 06:15 PST, Bruno) inputs and engine copied from sink-value
Copied from sink-value (oracle-1101-sink-value, `projects/sink-value/raw`, `src1025/`, `common.py`, `patch.py`), so
no job reads a closed room's folder; `raw/` and `src1025/raw/cache/` are not in git. sha256 (first 16 hex), equal to
sink-value D0 and massive-dims D0:
  - `src1025/raw/cache/base/ref_rec_tail.npy` 90a7f48055d7072d (monitor cue-on)
  - `src1025/raw/cache/base/none_rec_tail.npy` 773c280c146fe5b7 (monitor cue-off)
  - `raw/stack/cache/eager_rec_tail.npy` e1d6b75ca8b42b45 (grader cue-on); `compiled_rec_tail.npy` 773c280c146fe5b7
  - `raw/r50/runs/r50_s0_lr0.0001_kl1_at8192_merged.pt` 668be0454fec5c1c, `r50_s1_...` 9f9d286c81d6c7cc
  - `raw/stack/runs/stack_s0_lr0.0001_kl1_merged.pt` 5898b14ccde9458a, `stack_s1_...` b3eb8a05c81127d5
`ref/` holds sink-value's frozen.json, summary.json and all its patch files (the row-G targets). `patch.py` is
sink-value's engine plus three additions (kdims, alpha, mode "zero"; see its docstring); its old paths are unchanged.
`load_tail` casts the rotary inv_freq buffer to bf16 (as in every earlier run of these copies): pair 1 runs at 0.8047
instead of 0.8058 rad/token. pp.py asserts the pair map (cos[:, j] == cos[:, j+64]) and that speed falls with j.

## D1 (2026-11-03 06:12 PST, Felix) capture pass before the final freeze
Vera's capture-only screen (#13492, 06:00 PST) computed head 27.8's per-pair score split and the attention-only
prediction (band 0.115/0.128 of the score change; 9.7%/0.7% of the redirect) before PREREG v1.1 (s9) was frozen.
PREREG v1 (s5 status rule, nulls, thresholds; commit 30256602, 06:00:17) was set before it and was not changed.
v1.1 added only the G27.8 sink profile, r_S averaging, the point-estimate NO wording, exploratory TOP3 and the
one-job compute rule. Effect on the headline: none on the rule; the prediction in s4 is not blind.
