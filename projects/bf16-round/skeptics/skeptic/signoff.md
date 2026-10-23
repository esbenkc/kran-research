# Sign-off — Vera (skeptic), room oracle-1023-bf16-round

**Verdict on the title ("Does an explicit 16-bit round at the probe's split point make a compiled read-only probe
invisible to a trained detector?"): REJECTED (no)**, on the frozen stack: Qwen3-1.7B, bf16, Apple M4, torch 2.14.1
MPS, torch.compile (Inductor, Metal), probe = forward hook on block-14 `self_attn.q_norm` that keeps a reference.
Title arm = R2 `round16`: q_norm returns `(weight.float() * h.to(bf16).float()).to(bf16)` in both graphs.

Scope the post MUST keep: on a torch nightly (2.16.0.dev20261022, git ad8c2fc, which has PyTorch PR 200559) the same
line gives 0/64 changed prompts. So the "no" is a statement about torch 2.14.1's Metal code generator, not about
rounding in general.

I did not read build's code. Every number below comes from build's saved raw files or from my own model loads.

## Key numbers I rebuilt

| Claim | Build | Mine | Checker / output |
|---|---|---|---|
| Metal gate: round present in R2 no-probe source | yes | yes: `round16_noprobe_A.py:866-867` `tmp27 = static_cast<bfloat>(tmp26)`; stock feeds fp32 product to rope | read source; T1 review |
| naive cast / bit-view erased | yes | generated source == stock in 4/4 graphs after stripping comments | diff; T1 review |
| R0 stock: changed prompts /64, A+B x passes 0-3 | 64 | 64 everywhere, max abs last-logit change 1.1406 | vera_rebuild.py -> vera_rebuild_emu0.out |
| **R2 round16: changed /64** | **64** | **64 everywhere, max 0.8438; 77.67% of all logits** | vera_rebuild_emu0.out, vera_frac.out |
| intround (form B) | 64 | 64, and == round16 unprobed bit for bit | vera_rebuild_emu0.out |
| fresh detectors seeds 1 / 2 on R2 (post-result robustness, D3) | 71.1 / 91.4 | 0.7109 / 0.9141; AUROC 0.956 / 0.968; shift toward Yes 128/128 each. Seed 1 sits below 75% because it says Yes too often (unprobed->No 0.43), not because it cannot separate | vera_det_seeds12.out |
| barrier (opaque op, both graphs) | 0 | 0; barrier unprobed == stock probed (0/4544 rows) | vera_rebuild_emu0.out |
| R2f (round + 32-bit rope) | 0 | 0 (R0 64 in same process); R2f unprobed == stock probed | vera_rebuild_rest.out |
| emulate_precision_casts (none/cast/round16) | 0 | 0 | vera_rebuild_rest.out |
| eager stock / round16 | 0 | 0; eager round16 == eager stock | vera_rebuild_rest.out |
| nightly R0 / R2 | 64 / 0 | 64 / 0 | vera_rebuild_nightly.out |
| determinism floor (pass vs pass, A vs B) | 0 | 0 in every graph of every run | all vera_rebuild_*.out |
| SHA256 definition | - | recomputed == saved on 568/568 rows per graph | vera_rebuild.py |
| **fresh detector (seed 0, trained on R2) on R2, whole graph** | **91.0%** | **0.9102 (233/256) from raw json; my OWN run: answer logits bit-identical to build on 128/128 x 2** | vera_det.out, vera_round.py, vera_fresh_cmp.out |
| fresh detector on 128 test prompts never scored by build | - | **0.9102** (No 118/128, Yes 115/128); 256/256 prompts shift toward Yes | vera_fresh_cmp.out |
| fresh detector on stock | 71.9% | 0.7188 | vera_det.out |
| saved detectors s0/s1/s2 on R2 | 80.1/59.8/68.8 | 0.8008/0.5977/0.6875; AUROC 0.865/0.904/0.892; shift toward Yes 128/128 each | vera_det.out |
| saved detectors on emulate | 50.0 x3 | 0.5000 x3, 0/128 changed, every shift exactly 0 | vera_det.out |
| speed cost R2 vs stock | -0.4% (+0.4% in rerun) | -0.37% [-0.67,+0.28]; rerun +0.44% [-0.88,+0.98] | vera_timing.py -> vera_timing.out, vera_timing_all.out |

Frozen rule: R2 passes the Metal gate, changes >= 1 prompt (64/64), and the fresh detector scores >= 75%
(0.910, Wilson 95% about [0.87, 0.94] treating answers as independent; 466/512 = 0.910 with my extra 128 prompts).
So REJECTED.

## Null tests
1. Never-attached null detector (1022 recipe, labels on identical inputs): 0.4883 on stock and on round16,
   AUROC 0.501/0.484 (rebuilt from raw, vera_det.out).
2. Within-pair label swap, 2000 draws, on every detector file: 95th pct 0.51-0.57, max 0.52-0.66 vs fresh 0.910.
3. Held-out prompts: my own run on test0448-0575 (never scored by build) gives the same 0.9102.
4. Bit-level null: same build compiled twice (A vs B) and three warm passes: 0 changed prompts everywhere, so
   "changed" is not run-to-run noise.

## Definitions checked
- "Explicit 16-bit round" = R2 as frozen in PREREG.md before any comparison; it survives to the Metal source.
  It reproduces the probed value exactly: R2 probed == stock probed, 0/64.
- The leak under R2 is the next step's arithmetic: Metal types the rounded value as `bfloat`, so the no-probe rope
  runs `bfloat*bfloat` (round16_noprobe_A.py:873). R2f (same round, rope in 32-bit) gives 0/64. That is the
  mechanism sentence, not the title arm.
- "Trained detector": the fresh detector is trained on R2's own block-26 inputs (train split), scored on the whole
  compiled graph without a recorder. Its dev curve (0.859 at step 1024) is close to its test score, so no sign of
  test leakage. I could not verify the train/test split from code (I do not read build code).

## Head clause (prereg item 5, foldout) — definition stretched
Removing the top 1-3 heads gives exactly 50% accuracy in all 3 seeds, but this is a PINNED answer: the tail says
No (s0) or Yes (s1, s2) on 128/128 prompts in both conditions, while the probe still raises the Yes-minus-No
log-odds on 127-128 of 128 prompts (AUROC 0.60-0.93). The clause holds by accuracy only; the signal survives
(vera_heads.py -> vera_heads.out). The post must not say the heads carry the detection signal.

## Prereg outcome
FAILED: clause 1 (R2 0/64) fails with 64/64; clause 4 (emulate leaves >= 1 changed) fails with 0/64.
Clause 2 fails (s0 0.80 >= 0.60). Clause 3 (speed < 10%) holds. Clause 5 holds only in the degenerate sense above.

## Compute notes
vera_round.py (1.7B, MPS, compile, 256 test prompts x 2 conditions) ran alone in build's slot: peak footprint
5.91 GB for ~2.5 min (over the 5 GB line; nothing else ran). vera_heads.py CPU: 0.99 GB, next to build's 3.6 GB
training job (4.6 GB total).
