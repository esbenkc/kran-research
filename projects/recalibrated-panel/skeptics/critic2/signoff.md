# critic2 sign-off (final cross-family review, restarted on Claude)

The gpt-5.5 critic2 session gave a staleness-only REJECT (room #7959; its numeric rebuild matched) and then hit its
$2.25 cap. This restart ran on Claude (same vendor as the team, so this is not a cross-vendor check).

Verdict: PASS (status REJECTED reproduces). Read: blog PR #93 head c57adef, preview
https://deploy-preview-93--kranai.netlify.app/recalibrated-panel/ (caption 'except the last row' present = c57adef).

Own code: `c2_rebuild.py` (reads only data/ref/split.json, data/scores/*.npz, data/states/*/{kl,lp,off}.npy).
Output: `c2_rebuild.out`, `c2_rebuild.json`.

| cell | post | rebuild |
|---|---|---|
| Rule B per run (s0/s1/s2) | 43.6 / 17.8 / 37.6% | 43.6 / 17.8 / 37.6% |
| Rule B pooled, eval FA | 33.0%, - | 33.0%, 1.51% |
| logistic head pooled (FA) | 84% (58-98), 1.2% | 83.9% (97.7/58.2/95.8), 1.17% |
| QDA head pooled (FA) | 77% | 77.1%, 0.89% |
| shuffled-label logistic null (own rng seed 1) | - | 0.0-1.5% per run, 0.67% pooled |
| clean model Rule B | 99.5% | 99.5% (FA 1.3%) |
| no-reward-for-spread run Rule B / logistic | 3.7% / 60% | 3.69% / 59.7% |
| eval KL pooled (meta definition) | 0.0039 | 0.00393 |
| right-country pairs | 298/298 every run | 298/298 every run (summed statement log-prob) |

Caveats (do not change the headline):
1. KL: a plain mean over every token of the 1192 eval statements in states/*/kl.npy gives 0.00414 pooled
   (stop-grad 0.00706), vs the reported 0.00393 (0.00643) from meta.json `kl_eval_mean_every_position`.
   Both are far under the 0.0096 bar; the post uses the attack-time definition consistently.
2. Front matter `lab_cost_usd: 0.6` is below the room's model-API spend (question 0.51 + gpt-5.5 critic2 2.23 = 2.74).
3. sklearn on macOS Accelerate prints spurious matmul overflow warnings; results are finite and match ceiling.json.
