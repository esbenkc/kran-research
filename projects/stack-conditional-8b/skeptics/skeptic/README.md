# Vera (skeptic) receipts — stack-conditional-8b

Own code, written without reading analyze.py. Run from this folder after the job's data lands.

- `vera_check.py ../../data/eval --step 8192` — frozen status rule (PREREG s3) from the per-prompt whole-graph P(X)
  rows: paired dGAP, 2,000-resample prompt bootstrap, YES_k/NO_k, controls |dGAP| < 0.03, recorder-only (tail_eval),
  seed-level t-interval (reported), and a build-label shuffle null per run.
- `vera_onset.py ../../data/runs ../../data/t_pred.json ../../data/rows02_dev.npz` — s rebuilt from the raw bf16
  block-34 rows 0-2 and the layer-34 RMSNorm weight, T_naive / T_x33, T_obs per run, scored onset clause.
- `vera_06b_rule.py` — retro-check of the rule on the public 0.6B parent (room #31019).
- `vera_fixture.py` — 8 planted-edge cases (CONFIRMED / REJECTED / UNCONFIRMED edges at 0.10, 0.5x, 0.03,
  recorder-only); pre-data result: vera_check.py and analyze.py agree 8/8. Paths assume the room's work dir layout.
- `leak/` — cross-run leakage test of brl.py (stack:0 batched with penall:0 vs shuf:1): bit-identical.
- `novelty.md` — prior-work check.
- `vera_law.py`, `rep_analysis.json` — cue-size-law-8b-rep per-seed fits (from its public branch) for T at a share.
