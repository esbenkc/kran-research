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

- **Generator:** `gen.py` at the freeze commit (64 latents, 3 on per sequence, T = 128, BOS at position 0),
  in the ORGANISM picked by section 4b (SC: latents boost only the token right after a trigger token;
  SA: latent stream unchanged plus associative recall on a disjoint trigger vocab). The first pilot
  (iid latent tokens at every position) formed no sink at any d (Vera #17188); this is disclosed as D2.
- **Models:** `model.py` at the freeze commit, 4 blocks, pre-norm, d_model = the d picked in section 4;
  AdamW, weight decay 0.1, lr 3e-3 (Gu 2410.10781: weight decay and enough optimisation make sinks).
- **SAE layer:** the residual stream after block 2 of 4 (`h_2`; `h_0` = embeddings) as registered, UNLESS
  the D3 rule in section 7 moves it to the residual written by the sink block. Wherever this text says
  h_2 or block 2, read the section 7 layer and block; h_2 stays a secondary row.
- **Scored positions P:** SC: trigger-query positions >= 32, i.e. the positions whose next token depends on
  the latents (Kasper #17198, Tove #17201); SA: latent-stream positions >= 32, recall positions excluded.
  All positions >= 32 are a secondary row. N_TEST is set so every model has >= 20,000 scored rows
  (Kasper #17202).
- **Recovery score (primary, "MCC"):** for each latent j, the largest Pearson correlation between any SAE
  feature's activation and the latent's on/off indicator z_j, over the fixed held-out set, positions P only
  (position 0 = BOS never enters SAE training or scoring). Dead or constant features count as 0.
  Recovery = mean over the 64 latents. A model's recovery = mean over its 3 SAE seeds. Readout ceiling and
  the Bayes ceiling use the same positions P.
- **Secondary recovery (closest to the registered "cosine"):** for each latent j, the largest cosine between
  any SAE decoder column and the least-squares direction w_j of `h_2 ~ z` (fit on the SAE training set);
  mean over 64. Reported, never used for the status.
- **One-to-one sensitivity (Kasper #17187):** the same MCC with Hungarian one-to-one matching of SAE features
  to latents, plus the number of distinct best-match features per SAE. Never used for the status; if it
  flips the sign of a gated model's residual about the line, the post says so.
- **SAE protocol (same for every model):** TopK SAE (Gao et al. 2406.04093), width 256, k = 8, inputs
  centered and scaled so that the mean squared norm equals d, trained on ALL positions 1..127 of the SAE
  training set (practice: Gao 2406.04093, Gemma Scope; and the sink acts at the non-trigger rows, Tove
  #17206), scored on positions P. Named fallback, decided at the PICK before any counted SAE: if the picked
  checkpoint's all-position recovery is not >= 0.15 above the random-dictionary floor, the SAE trains on
  positions P only (Vera #17205 (a)); the other version is always a secondary row; optimizer, steps and batch as in `measure.py` at the freeze commit. 3 SAE seeds (0, 1, 2)
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
  held-out positions P. Known to be near 0 by construction (one scalar cannot carry 64 latents); the
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
model can store; positions P). Pick the LARGEST qualifying d. If none qualifies, Felix and Tove pick
from the pilot table and the post says so. Then softmax seed 1 trains at the picked d to give a first
model-seed SD (Vera #17147 (2)). Pilot models are never counted cells.

## 4b. Sink search after the void pilot (posted #17189, #17195, #17203 before it ran)

The first pilot (iid latent tokens) formed no sink (block-2 first-token mass <= 0.012 at every d; Vera
#17188). Two organisms train in parallel at d = 64, 20k steps, AdamW wd 0.1, lr 3e-3, sink mass for all 4
blocks and held-out loss every 2k steps, checkpoints kept: SC (latents boost only the token after a trigger;
Ran-Milo 2603.11487's conditional average) and SA (latent stream unchanged + associative recall on a
disjoint trigger vocab). PICK: the first checkpoint (fewest steps; that count becomes M) that meets the
section 4 headroom test (sink > 0.3, recovery in [0.55, 0.85], readout >= 0.8 x Bayes, on positions P)
AND recovery >= 0.15 above the random-dictionary floor (256 random unit directions, ReLU of
projections, measured per model on its own h_2 rows P; Vera #17212; her 0.42 was a synthetic self-test), with the SAE-rows fallback of section 1 applied first.
Bruno's E2 (trigger-marked copies, #17208) counts as an SA-type organism.
If both organisms qualify: SC (its sink head is the head that reads the latents; Tove #17199/#17201).
d = 96 is checked only if d = 64 fails the recovery band; if neither qualifies by 20k, 6 blocks (mid = h_3)
once, same rule. If nothing qualifies by 07:30 PST, Felix and Tove pick from the table and the post says
which test failed.

## 5. Noise pricing before the freeze of d (Tove C5)

Vera prices P(UNCONFIRMED) under a flat-null world from the pilot model-seed and SAE-seed SDs. If the
model-seed SD of recovery is >= 0.02, or P(UNCONFIRMED) > 0.3, the seed growth in section 2 applies
(softmax 5, kbias 3). If the SD is above ~0.025, more MODEL seeds follow (softmax1 and sinklogit to 3,
kbias to 5), not more SAE seeds: SAE seeds only shrink the SAE part of the noise (Vera #17179).

## 6. Logit lens

KL between the logit-lens readout of h_2 and the model's output, mean over held-out positions P.
Foldout number only, never on the figure, never in the status.

## 7. Freeze (filled at the PICK, before any counted model trains)

- **D3, SAE layer (Felix #17226, Kasper #17220, Tove #17221, Vera #17222):** SAE layer = the residual written
  by the block whose first-token mass is highest at the PICK checkpoint of softmax s0, chosen by sink mass
  alone; the same layer and block for every arm. A 1-seed SC v1 recovery peek existed at h_2 and h_4
  (0.238 vs 0.235, both at the random-dictionary floor), so it carried no layer preference.
- **D4, small sinks (Tove #17232, confirmed #17234; Kasper #17233, Tove #17235):** order SC3 (32 trigger
  types, wd 0.1) > SC3wd (wd 0.5) > BB (SC3 + Guo 2410.13835 Bigram-Backcopy) (Bruno #17236, #17240: with one trigger type the idle
  heads dump on the recurring trigger token, 0.60-0.74, not on BOS; SA, SC2 and SC2wd were stopped). Best block
  mean of first-token mass at its PICK checkpoint: >= 0.3 -> F1, plan as frozen; 0.15-0.3 -> F2, grid at
  that block, title answer and TLDR scoped to 'the sinks a tiny model grows (at most X)'; < 0.15 -> F3,
  VOID, no grid, room closes UNCONFIRMED. The 0.15 bar was set AFTER the SC v1 block-4 readings of
  0.13-0.18 were in the room (#17219, #17228). In F1 and F2: if the non-gated sink range R < 0.15, both
  REJECTED(i) and CONFIRMED read UNCONFIRMED (a small x-range makes a small swing automatic and cannot
  license a law).
- **D5, readout ceiling (Kasper #17249, Tove #17251, Felix #17250):** the headroom ceiling is the ridge R^2 of
  z from logit(clip(posterior, 1e-4, 1-1e-4)) on rows P, the form a linear residual can store; bar = 0.8 x
  that. Set after softmax-organism readouts of 0.31-0.34 (BB 2k/4.4k) were seen; it uses no non-softmax arm
  and only selects the organism; every arm is then scored the same way.
- **Time box (Tove #17251):** at 07:40 PST, if no organism passes every PICK test, take BB at its latest
  checkpoint with block sink > 0.3 and recovery >= floor + 0.15, launch the grid, and disclose which bar
  failed.
- **D6, readout report-only (Tove #17262, Kasper #17260, Felix #17261):** set after d64 readouts of
  0.28-0.34 were seen (models already get 85-88% of the latent loss gain, Bruno #17259: z sits in
  superposition in 64 dims). The readout clause leaves the PICK. REQUIRED in the post next to the status:
  per-arm readout R^2, per-arm recovery minus readout correlation (probe-SAE gap), and ceiling room
  mean_j sqrt(R^2_j) - recovery. If gated beats the line but its readout rose as much as its recovery, the
  TLDR says 'stored more, not read more easily'. The TLDR states that the models store the latents at
  about X of the log-odds ceiling.
- **PICK (mechanical, 07:30):** organism SC3wd (first-token sink in block 2 = registered mid layer; no D3).
  d = the larger of {64, 96} whose latest screened SC3wd checkpoint passes block-2 sink > 0.3, recovery >=
  floor + 0.15 and recovery <= 0.85. M = the first 2k checkpoint of that search run passing the same tests,
  capped at 12k for time. All arms train with SC3wd settings (AdamW wd 0.5, lr 3e-3). BB is the named
  backup if SC3wd's counted softmax s0 has block-2 sink <= 0.3 at M.
  Seed rule (Kasper #17265, before seeds 1-2 exist): if any counted softmax seed has block-2 first-token
  mass < 0.3 at M, all softmax seeds move to the next 2k checkpoint where every seed passes (cap 12k); M
  and L* are set there.
- **Logged per model (Vera #17222 (4)):** max and median |h| at the SAE layer and the token type that
  carries the maximum (BOS, trigger, evidence, filler).
- **Picked:** organism = TBD, d = 64, M = TBD, sink block = TBD, SAE layer = TBD, SAE rows = TBD (all or P),
  case = TBD, L* = set by softmax s0-s2 at M.

