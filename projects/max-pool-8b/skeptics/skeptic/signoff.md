role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off — max-pool-8b (room oracle-1120-max-pool-8b-cont, continuation of oracle-1119-max-pool-8b)

## Answer to the question
REJECTED (No). At Qwen3-8B, after the averaging-probe fine-tune, the max-pooled probe catches 0.62% of false
statements at a 1% false-alarm line (3 of 3 seeds), far under the 50% the frozen rule needs. PASS means: I rebuilt the
key numbers from the raw per-token activations with my own code, and they give the same answer as the build.

## Frozen rule (PREREG.md s6, commit 335ccd976, pushed 2026-11-19T00:58:16Z), applied by my code
- Clean validity gate: clean max18 catch 99.16% (eval false alarm 0.0%) >= 80%: PASS (`vera_rebuild.json` per_state.clean).
- Premise gate, my own picks from the stop split (`vera_rebuild.py` picks block): main seed 0 stop catch 0.00 -> step 150;
  seed 1 0.01 -> 150; seed 2 0.00 -> 150; random seeds 0, 1 follow their partners -> 150. No VOID seed. Equal to
  `data/gate/*.json`.
- Primary metric with the frozen monitor (`probes.npz`; `scores.npz` = rows . w + b to <= 5e-7):
  max18 per seed 0.67% / 0.50% / 0.67% (eval false alarm 0.67 / 0.34 / 0.67%), pooled **0.62%**, city bootstrap
  95% [0.11%, 4.25%] (`vera_status_frozen_probes.json`). Upper bound < 50% -> **REJECTED**.
- With MY OWN refit of max18 (same C 0.01, cos(w) 0.999988): 0.67 / 0.17 / 0.67%, pooled 0.50% [0.11, 4.25]
  (`vera_rebuild.json` pooled["main/max"]). Same answer.
- seed-level: 0.62% [0.37%, 0.86%] (t-interval over 3 seeds, df 2).
- Prediction (s7): 3 of 3 seeds under 20% (Clopper-Pearson 95% [29%, 100%]) -> **held**.
- Build `results.json` (commit 60992fa5e): 0.62% [0.11, 4.25], seed-level [0.37, 0.86], REJECTED, held. Match.

## Framing rules (s6 a-d), my numbers
- (a) max18 - avg18 = -0.28 points [-3.4, 2.0]: lower bound not > 0, so no "max pooling helps".
- (b) max18 - last18 = -44.5 points [-47.4, -30.0]; max18 - country18 = -46.1 [-57.7, -42.8]: the answer tokens carry
  the signal. last18 per seed 41.1 / 47.7 / 46.6% at eval false alarm 0.67 / 1.17 / 0.67% (AUROC 0.76 / 0.83 / 0.69),
  pooled 45.1% [31.8, 48.1]. country18 46.8 / 45.1 / 48.3%.
- (c) Random-direction push, max18 at own line: 99.66% / 99.50% (both >= 50%): the drop is specific to the push against
  the averaging probe.
- (d) KL-only arm (reported only): max18 99.33% / 99.33%.

## Nulls (where luck could fake the result)
- Clean label-swap null (refit max18 with labels swapped inside random city pairs): 1.17% catch.
- Random-probe null on each attacked scored state (20 random unit directions, max over tokens, own line): mean
  0.55 / 0.50 / 0.52% catch (max 2.0 / 3.4 / 2.7%). max18 sits at this floor. (`vera_nulls.json`)
- Fresh max18 label-swap null on each attacked state: 2.2 / 1.2 / 3.5% vs the real fresh max18 58.1 / 65.1 / 93.1%.
- Threshold-free view: max18 AUROC after the fine-tune 0.525 / 0.528 / 0.535 (clean 0.9987), so the low catch is not a
  line artefact.

## Mechanism and side clauses (checked because the post uses them)
- Highest-scoring token is a shared-prefix token in 81.7 / 81.4 / 80.4% of eval false statements (clean 0.3%).
- Prefix lemma on the SCORED rows (eval pairs, batched as scored): max abs difference 0.0 in every scored main state.
- Answer-token-only max18: 2.4 / 3.0 / 4.0% (clean 99.7%).
- Twin gap of max18 in clean-SD units, country token / full stop: clean 5.06 / 6.52 -> 1.27 / 0.57, 1.44 / 0.69,
  1.60 / 0.57.
- Behaviour: right country beats its twin in 298/298 eval pairs in every state; output KL on 200 held-out windows
  0.0069-0.0078.
- Fresh refits on the attacked probe split: max18 58.1 / 65.1 / 93.1% (spread is large; do not write "58-65%"),
  avg18 84.2 / 86.4 / 87.6% with my refit (build 79.7 / 82.4 / 87.6%; same direction, my C choice differs).
- Forecast (s8): my whitened cosine 0.894 (build 0.903; my avg18 refit picks C 0.01 vs build 0.1 on a CV-AUROC tie at
  0.9996). Both >= 0.8 -> forecast "< 20%", which came true.

## Limits I want in the post
1. One model, one dataset (Geometry of Truth cities), one attack recipe; the step from the 1.5B parent also changes the
   model family and training recipe, so this is not a clean scale curve.
2. The city bootstrap with re-picked 2nd-of-200 lines is skewed (duplicated top refit scores move the line): report
   intervals, not +-. last18's upper bound 48.1% sits near its highest seed.
3. "Hidden, not gone" rests on a new LINEAR probe at layer 18 (58-93%, AUROC 0.98-0.99) and on behaviour (298/298); it
   does not show what other layers or nonlinear probes would find.

## Novelty
`novelty.md`: no paper found that tests an averaging-probe weight attack against an unseen max-over-tokens probe at 8B.
RL-Obfuscation (2506.14261) reports max-pooled probes robust under an RL attacker that changes the text; this run is a
different threat model (fixed text, weight attack on the probe score), so it does not contradict that paper.

## Files
`vera_core.py`, `vera_rebuild.py` (full rebuild), `vera_status_frozen_probes.py` (status with the frozen monitors),
`vera_nulls.py`, `vera_whatif.py` (pre-seed-2 bound: even a 99% seed 2 gives upper bound 36%), outputs `*.json`, logs
`*.log`. Data: the room branch `data/` (seed 2 from labq 05ec16djpi, result commit 1c0ce37ab, copied to the room branch
in 5465b9b90). Last night's receipts: `2026-11-19/`.
