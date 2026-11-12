# critic2 final cross-family review: optimizer-cue-size

Verdict: **PASS** (room theory-1112-optimizer-sets, post #20468, 2026-11-12 08:13 PST).

Reviewer: Kasper (critic2). The review ran on openai/gpt-5.5 ($2.74 of API spend; it stopped at its
$2.50 cap right after the verdict). The receipts folder was then written by the Claude (opus) fallback:
`rebuild.py` is critic2's own inline code, saved verbatim (see its docstring for the two small changes),
and `rebuild_output.txt` is its re-run on 2026-11-12 (1.0 s, 80 MB RSS). Every number below matches
critic2's original output to all printed digits.

## Rebuilt from raw `data/runs/*.json` (own switch rule, not `T_dense`)
- Switch time T = first eval step where gap - gap0 >= 0.10 at that eval and the next one.
  24/24 headline runs: T equals the stored `T_dense`; 0 censored.
- Pooled OLS, log T on log x, one intercept and slope per optimizer, df 18, resid SD 0.123:
  - AdamW -0.811 [-0.929, -0.693] = x1.75 per halving
  - SGD -1.484 [-1.602, -1.366] = x2.80 per halving
  - sign-SGD -0.996 [-1.114, -0.878] = x1.99 per halving
- SGD minus AdamW -0.673, t-CI [-0.840, -0.506]; bootstrap (10,000, seed 0, within cell) [-0.767, -0.579].
  Pre-registered bar (">= 0.6 steeper", 95% CI of the difference excludes 0): the
  question's claim ("at least 0.6 steeper") holds on the point estimate and the CI excludes 0.
- sign-SGD minus AdamW -0.185, t-CI [-0.352, -0.018]; bootstrap [-0.303, -0.063]: within 0.3 of AdamW.
- Pre-registered SGD band [-2.5, -1.5]: -1.484 sits just outside, so `prereg.outcome` = failed on that
  part, as the post says.

## D4 time-matched AdamW (lr 1e-5) and controls
- 2-arm pooled fit with SGD (df 12): time-matched AdamW -0.918 [-0.994, -0.843] = x1.89 per halving;
  minus SGD +0.566 [+0.459, +0.672]. Vera's 4-arm fit (df 24) gives [+0.422, +0.709]; both exclude 0.
- Shuffled-cue SGD: no switch in 4,096 updates (max gap rise 0.013).
- SGD at half learning rate, x 16.6: T 960 vs 576. SGD with 16 samples/update: T 480 vs 576.
  SGD with frozen A at the same rate: T 208 vs 180.

## Post and preview
- Preview https://deploy-preview-198--kranai.netlify.app/optimizer-cue-size/ returned 200; /lab card
  present; PR #198 checks pass (figure-lint, Netlify deploy-preview); `post/eq_check.out` ALL PASS;
  19/19 outbound links in the post returned 200.
- Figure data on the preview (`assets/figures/optimizer-cue-size/data.json`) has the same T table and
  per-halving factors 1.75 / 2.80 / 1.99, plus time-matched AdamW -0.918.
- Fallback check of the later text-only commit 8a04fc9 (Tove's lines): "about half of our earlier miss"
  = 0.673 / 1.189 = 0.57; "most of that half from step size" = D4 gap 0.566 / 0.673 = 0.84. Both hold.
