# PREREG addendum v1: gated-sinks (theory-1108-gated-sinks)

The registered prediction in the room goal (fixed 2026-11-08 14:52 CET, #17108) is never edited.
This addendum only makes it measurable and adds the law test that Tove approved (#17129, C1-C5).
Two steps: this text is reviewed now (before the pilot). The freeze is the commit that adds the picked d
(section 4) next to `gen.py`, `model.py`, `train.py` and `measure.py`; Felix posts its hash. No counted run
(B2) starts before that hash is posted. Any later change is a numbered deviation (D2, D3, ...) posted in
the room and disclosed in the post.

## 0. Registered prediction (copied, unchanged)

> At matched loss, the gated models' mean SAE latent recovery is >= 0.10 higher (absolute, cosine) than
> softmax in 3 of 3 seed pairs. The gated model's first-token sink mass is < 20% of the softmax model's.
> The norm-scale check shows no compensating hiding (latent decodability from per-token norm < 0.2 R^2 in
> both). P = 0.35.

## 1. D1: operationalisation of the registered words

The generator latents are not directions in the residual stream, so "max cosine to true latents" has no
direct meaning. D1 replaces it as follows (disclosed in the post as D1).

- **Generator:** `gen.py` at the freeze commit (64 latents, 3 on per sequence, vocab 128, T = 128, BOS at
  position 0, repeated spans in half the sequences).
- **Models:** `model.py` at the freeze commit, 4 blocks, pre-norm, d_model = the d picked in section 4.
- **Mid layer:** the residual stream after block 2 of 4 (`h_2`; `h_0` = embeddings).
- **Recovery score (primary, "MCC"):** for each latent j, the largest Pearson correlation between any SAE
  feature's activation and the latent's on/off indicator z_j, over the fixed held-out set, positions 32..127
  only (position 0 = BOS never enters SAE training or scoring). Dead or constant features count as 0.
  Recovery = mean over the 64 latents. A model's recovery = mean over its 3 SAE seeds.
- **Secondary recovery (closest to the registered "cosine"):** for each latent j, the largest cosine between
  any SAE decoder column and the least-squares direction w_j of `h_2 ~ z` (fit on the SAE training set);
  mean over 64. Reported, never used for the status.
- **SAE protocol (same for every model):** TopK SAE (Gao et al. 2406.04093), width 256, k = 8, inputs
  centered and scaled so that the mean squared norm equals d, trained on positions 1..127 of the SAE
  training set; optimizer, steps and batch as in `measure.py` at the freeze commit. 3 SAE seeds (0, 1, 2)
  per model. k = 4 and k = 16 run as a sensitivity row only.
- **Sink mass (x-axis, primary):** mean attention weight on position 0, block 2's attention, over all heads,
  query positions 8..127, over the held-out set. Secondary: the same averaged over all 4 blocks; and
  "dump mass" = position 0 plus the learned sink slot (softmax1 / sinklogit / kbias null slot).
- **Matched loss:** the 3 softmax seeds train for M steps (M in `train.py` at the freeze commit). Target
  L* = the largest of their held-out losses at step M. Every model (softmax included) is analysed at its
  first eval whose held-out loss <= L*. Evals run often enough that each analysed checkpoint has
  (L* - L)/L* <= 0.01 of TOTAL loss; `measure.py` asserts it per model. If a model overshoots, it is
  retrained from the last checkpoint above L* with a 10x finer eval grid; if it still overshoots, it is
  dropped and disclosed. L* is fixed before any non-softmax arm trains. A model that does not reach L* in
  3M steps is dropped and disclosed. The post reports, per model, the gap to L* both as a share of total
  loss and as a share of the excess loss over the Bayes-optimal loss of the generator.
- **Training length (Kasper #17157 (3)):** matched-loss stopping gives each arm its own step count, and
  sinks grow with training (Gu 2410.10781). results.json logs steps per analysed model; the post reports
  Pearson(steps, sink mass) over the non-gated fit. If |r| > 0.8, the post says that sink mass and step
  count cannot be separated in this grid.
- **Registered norm check:** ridge R^2 of z_j from the per-token norm of h_2 (one scalar), mean over 64,
  held-out positions 32..127. Known to be near 0 by construction (one scalar cannot carry 64 latents); the
  post says so.
- **Live hiding checks (secondary):** (a) readout ceiling: ridge R^2 of z from h_2 per model; (b) gated
  models: ridge R^2 of z from block 2's gate vector vs from h_2; (c) SAE recovery on per-token
  RMS-normalised h_2 vs raw h_2 (SoLU hid features as small activations that a later norm rescales; Iris
  #17142); (d) per-latent gap = readout correlation minus SAE best-match correlation. If (b) gate R^2 >
  residual R^2, the latents moved into the gate (the SoLU case).
- **Floors (controls):** the same SAE protocol on h_0 (embeddings) and on h_2 of an untrained model of each
  arm (seed 0). The post reports recovery next to these floors.
- **Logged per model (Kasper #17182):** parameter count (gated adds a d x d gate per block), dead-feature
  share per SAE, steps to L*. If the mean dead-feature share differs between arms by > 10 points, the post
  says so next to the result. The Bayes posterior used for the headroom ceiling skips copied positions
  (repeat spans), where tokens are not i.i.d. given z.

**Registered prediction outcome (prereg.outcome):** `held` iff all three hold: (a) recovery(gated s) -
recovery(softmax s) >= 0.10 for s = 0, 1, 2; (b) sink(gated s) < 0.2 x sink(softmax s) for s = 0, 1, 2;
(c) registered norm R^2 < 0.2 in all 6 main models. Otherwise `failed`. (`unsettled` only if a main model
is dropped.)

## 2. Arms and grid

| arm | what changes | seeds | role |
|---|---|---|---|
| softmax | standard causal softmax attention | 0, 1, 2 | line fit; sets L* |
| softmax1 | softmax with +1 in the denominator (fixed null logit 0; Miller 2023 "Attention Is Off By One", Kaul et al. 2410.17174) | 0, 1 | line fit |
| sinklogit | learned per-head null logit in the denominator | 0, 1 | line fit |
| kbias | learned per-head key k* with value 0 (Gu et al. 2410.10781, Tab. 4) | 0, 1 | line fit |
| gated | elementwise sigmoid gate on the attention output (2505.06708, G1) | 0, 1, 2 | predicted |

No gated arm enters the line fit. If the pilot model-seed SD of recovery is >= 0.02 (section 5), softmax
grows to 5 seeds and kbias to 3 seeds (the two ends of the line) before anything else changes. If time
breaks the plan, softmax1 and sinklogit drop to 1 seed each first (disclosed); the 3 gated and 3 softmax
seeds stay.

## 3. The law test and the status (answers the title)

Hypothesis: less first-token sink mass means higher recovery, and sink mass alone sets recovery, whatever
removed the sink.

- **Line:** ordinary least squares, recovery = a + b x sink mass, on the n non-gated models (model means
  over SAE seeds; n = 9 in the base grid).
- **Slope CI:** the OLS t-interval, 95%, n - 2 degrees of freedom (Vera's pricing, #17147, used the same).
- **s_tot:** residual SD of the n non-gated model means about the line (n - 2 dof). It holds model-seed and
  SAE-seed noise together (Vera #17147 (1)). s_SAE (pooled SAE-seed SD) is reported beside it.
- **Band for gated model g:** band_g = max(0.03, 2 s_tot, h_g), where h_g = half-width of the OLS 95%
  confidence interval of the line's mean prediction at g's sink mass.
- **In band:** |recovery(g) - (a + b x sink(g))| <= band_g.
- **Swing:** S = |b| x R, R = max - min sink mass over the n non-gated models; its upper end =
  max(|b_lo|, |b_hi|) x R from the slope CI.
- **Sink levels:** the number of distinct sink levels in the fit = clusters of non-gated model sink masses
  where neighbours differ by > 0.1. If fewer than 3, the post calls the slope a two-group contrast (kbias
  vs softmax, gated vs kbias), not a 'law' (Tove #17137). Status rule unchanged; wording ceiling only.

**Status:**
- **CONFIRMED (yes):** the 95% CI of b lies entirely below 0 AND at least 2 of 3 gated models are in band.
- **REJECTED (no):** (i) the upper end of the swing is below a FIXED 0.05 (half the registered +0.10 gain; Tove #17161) (sink mass does not
  move recovery), OR (ii) at least 2 of 3 gated models are outside their band (gating does something that
  sink mass does not predict), OR (iii) the 95% CI of b lies entirely above 0 (more sink, better recovery).
- **UNCONFIRMED:** anything else (e.g. slope CI covers 0, swing not provably small, gated in band).

Precedence (Kasper #17157 (1)): REJECTED is checked first. A slope whose CI excludes 0 but whose swing is
below the bar is REJECTED(i), not CONFIRMED: a real but negligible slope is a no. A flat line never
confirms. If several REJECTED clauses fire, the post reports all of them.

Sign of the gated miss (Kasper #17157 (4)): results.json records, per gated model, the signed residual
recovery(g) - prediction(g). If (ii) fires, the post says "gating helps beyond its sink mass" (residuals
above the line) or "gating helps less than its sink mass predicts" (below).

## 4. Headroom rule (posted in the room before the pilot, #17131; Tove C2)

Pilot: softmax, seed 0, d_model in {32, 48, 64, 96}, 64 latents, trained M steps; 3 SAE seeds with the
protocol above. A d qualifies if: sink mass (primary) > 0.3, recovery in [0.55, 0.85], and readout ceiling
ridge R^2 >= 0.8 x the ridge R^2 of z from the exact Bayes posterior P(z_j | tokens so far) (the best any
model can store; positions 32..127). Pick the LARGEST qualifying d. If none qualifies, Felix and Tove pick
from the pilot table and the post says so. Then softmax seed 1 trains at the picked d to give a first
model-seed SD (Vera #17147 (2)). Pilot models are never counted cells.

## 5. Noise pricing before the freeze of d (Tove C5)

Vera prices P(UNCONFIRMED) under a flat-null world from the pilot model-seed and SAE-seed SDs. If the
model-seed SD of recovery is >= 0.02, or P(UNCONFIRMED) > 0.3, the seed growth in section 2 applies
(softmax 5, kbias 3). If the SD is above ~0.025, more MODEL seeds follow (softmax1 and sinklogit to 3,
kbias to 5), not more SAE seeds: SAE seeds only shrink the SAE part of the noise (Vera #17179).

## 6. Logit lens

KL between the logit-lens readout of h_2 and the model's output, mean over held-out positions 32..127.
Foldout number only, never on the figure, never in the status.
