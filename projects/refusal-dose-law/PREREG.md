# PREREG v1 — refusal-dose-law

Frozen by Felix (plan) before any TEST-harmful reply, judge label or first-token score exists. Git commit time is the
record. Room: theory-1031-refusal-dose-law. Sources: Tove APPROVE #10906, #10903, #10910, #10915, #10919;
Kasper #10901, #10907, #10913; Vera #10898, #10909, #10916, #10918; Iris #10900, #10905, #10912; Bruno #10895, #10902;
Selma #10899; Felix #10904.
The scout's pre-registered prediction (goal, fixed 2026-10-31 12:33 CET) stays as written. Its four clauses are
scored on their own below (prereg.outcome); the STATUS answers the title. Amend only by appending a dated "D<n>" entry
to DEVIATIONS.md.

## Title (yes/no)
Does the push along the refusal direction alone predict how strongly a single neuron pin bypasses refusal?

## Model, data, decoding
- Qwen/Qwen3-1.7B (HF cache), bf16, torch MPS, chat template with `enable_thinking=False`, no system prompt.
- Prompts: refusal-neuron room splits.json (copied to `data/splits.json`). DEV = 100 harmful (`dev_harmful`),
  TEST = 200 harmful (`test_harmful`, fixed order), TEST100 = the first 100 of TEST. No other split is scored.
- Greedy decoding, 64 new tokens. Every edit acts on the residual stream at the OUTPUT of layer 13 (the gate's
  write point), at every prompt token except token 0 (`<|im_start|>`) and at every generated token.

## The refusal direction r (one recipe; every number in the post uses it)
r = mean residual at the layer-13 output, last prompt token, over `dev_harmful[:100]` minus the same over
`test_harmless[:100]` (Vera's vera_dom.py recipe). r_hat = r/||r||. Bruno computes r with his own code and posts
cos(r_Bruno, r_Vera) before any TEST number; it must be >= 0.99 (bf16 vs fp32), else he stops and posts.
Kazemi et al.'s |cos| = 0.186 (2605.08513 App. G, their Arditi-selected r) is cited as theirs and never mixed into
our shares or p.

## Units and edits (per token t; a_t = the clean activation of the pinned neuron at token t, computed online)
- Gate G = layer-13 MLP neuron 3270, w = W_down[:, 3270]. Pin m: delta_t = (m - a_t) w. (signed; at small m some
  tokens get a negative edit, and p keeps that sign.)
- Pins: m in M5 = {6, 12, 18, 24, 30}. "Gate norm profile" n_t(m) = |m - a_t^gate| * ||w||.
- G_on (projection only): delta_t = (m - a_t)(w . r_hat) r_hat.  G_off (remainder only): delta_t = (m - a_t)(w - (w . r_hat) r_hat).
  G_on + G_off = G exactly.
- Dialed directions D(c, u): v = s c r_hat + sqrt(1 - c^2) u, u a random unit vector orthogonal to r_hat (seeded),
  s = sign(w . r_hat) (pushes the same way as the gate), delta_t = n_t(m) v. c in {0.05, 0.1, 0.2, 0.6, 1.0}
  (1 u-seed each) plus the TWIN c = |cos(w, r_hat)| (3 u-seeds): same p and same total push as the gate.
- Random neurons RN: 100 random layer-13 neurons j != 3270 (seed 0), at matched push: delta_t = n_t(m) w_j/||w_j||.
- Random directions RD: 20 random unit vectors in R^2048 (seed 1), delta_t = n_t(m) u.
- Natural neurons NN (held out, "found without search"): the 4 most negative-cos neurons 1265, 3990, 4786, 2142 and
  the 2 most positive 5861, 2591 (Vera #10909; Bruno confirms the list with r_Bruno before any TEST number), native
  pin delta_t = (m - a_t^j) w_j at m in M5. Positive-cos neurons test the other side: the curve predicts compliance
  BELOW base for them.
- Fake-r control FK: v = r_fake_hat, r_fake = DoM of two random halves (seed 2) of `test_harmless[100:200]` at the
  same layer and token, delta_t = n_t(m) v.
- p-matched random neurons PM (row only): 3 random RN neurons, pin scale chosen so their p equals the gate's p at
  m = 18 and m = 30.

## The dose number p (computed from the clean pass before any pinned run)
p(unit, m) = mean over TEST prompts and over prompt tokens t >= 1 of  -(delta_t . r_hat) / ||h_t||,
h_t = the clean residual at the layer-13 output (pre-RMSNorm). Positive p = push against refusal.
Pure dose d(unit, m) = the same mean of ||delta_t|| / ||h_t||. Row: p_last = p at the last prompt token only.
On-axis push in Arditi units (secondary): u(unit, m) = mean over TEST prompts of -(delta_last . r_hat) / ||r||.

## Labels
- Judge: 3 labels per reply, COMPLY / REFUSE / INCOHERENT, temperature 0, cached. Model claude-haiku-4-5 unless Bruno
  posts, before the first TEST call, a cheaper judge with >= 92% agreement with the last room's Haiku labels on its
  400 cached replies (base + pin 30); the model and the prompt sha are posted in the room before the first TEST call.
- Compliance (primary) = share COMPLY. Non-refusal (second labeling) = share not REFUSE.
- EFFECT(arm) = compliance(arm) - compliance(base) on the same prompts.
- First-token score (cheap, all units): the last room's first-token refusal logit difference at the first reply token;
  in prose only "the odds that the reply opens with I or Sorry", never "bypass".

## Judged cells (generation + judge), in run order; anything not done at 08:00 PDT is reported "not run"
1. base (TEST 200); G x M5 (TEST 200); G_on x M5 (TEST 200); G_off x M5 (TEST 200).
2. TWIN x 3 seeds x M5 (TEST100); D(c) for c in {0.05, 0.1, 0.2, 0.6, 1.0} x M5 (TEST100).
3. RN: first 10 neurons x {18, 30} (first 50 of TEST); RD: first 5 x {18, 30} (first 50 of TEST).
4. First-token score: 100 RN + gate at m = 30 (TEST100) (Spearman clause).
5. NN: 6 neurons x M5 (first 50 of TEST).
6. FK x M5 (first 50); PM: 3 x {18, 30} (first 50).
7. First-token score for all RN and RD at all M5 (TEST100), then the downstream-layer row (below).

## The curve
compliance(p) = sigmoid(alpha + beta p), 2 parameters, binomial maximum likelihood over individual replies.
FIT SET = TWIN + D(c) + RN + RD judged cells. Never the gate, G_on, G_off, NN, FK or PM.
Literal-scout variant (reported): the same fit on RN + RD only.
Same fit for the non-refusal labeling.

## STATUS rule (answers the title; mechanical)
gap(m) = compliance(G, m) - curve(p(G, m)), percentage points. 95% CI by bootstrap: 2,000 resamples, seed 0,
resampling prompts (paired across arms that share prompts) and refitting the curve each time.
- a_yes: at >= 4 of the 5 pins, |gap| <= 10 (point) AND the CI lies inside [-15, +15].
- a_no:  at >= 2 pins, |gap| > 10 (point) AND the CI excludes 0.
- b_yes: at m = 30, |EFFECT(G_on) - EFFECT(G)| <= 10 AND EFFECT(G_off) < 0.25 EFFECT(G) (points).
- b_no:  at m = 30, CI lower of EFFECT(G_off)/EFFECT(G) >= 0.25, OR |EFFECT(G_on) - EFFECT(G)| > 10 with CI excluding 0.
- CONFIRMED iff a_yes AND b_yes AND not a_no AND not b_no.  REJECTED iff a_no OR b_no.  Else UNCONFIRMED.
- Gate check first: EFFECT(G, 30) >= 20 points, else UNCONFIRMED.
- The status is computed with compliance AND with non-refusal. If the two disagree, the status is UNCONFIRMED.
- Wording frozen now: if write-layer p fails and the downstream row fits, the post says No and names propagation
  (Liu 2606.19831 Sec 8) as the candidate route. The downstream row never rescues the title.

## Scout prediction clauses (prereg.outcome: held = all four)
1. The curve (literal-scout fit, RN + RD only; the FIT SET version reported next to it) predicts the gate's judged
   compliance within 10 points at all 5 pins.
2. R^2 >= 0.8 across units: unweighted R^2 of cell compliance vs curve over the FIT SET cells (also on dialed only).
3. Spearman(|cos(w_j, r)|, |first-token score change|) >= 0.7 over the 101 neurons (gate + 100 RN) at matched push,
   m = 30. Scored on the first-token score because 101 judged units do not fit the clock (stated in the post);
   split-half reliability of the score reported first; the signed version reported next to it.
4. Pure dose: R^2 < 0.3 for sigmoid(alpha + beta d) on the FIT SET cells.

## Secondary (reported, not status)
- Arditi-unit midpoint (Iris #10912, Tove #10919): held iff the curve, re-expressed in u, reaches halfway from base
  compliance to 1.0 at u in [0.25, 0.75].
- Two-sided: NN positive-cos neurons below base compliance at m = 30 (CI upper < 0), and the curve's prediction for
  each NN cell; NN mean |observed - predicted| reported.
- Twin vs gate at each pin (same p, same push, different orthogonal part).
- Downstream row: layer L* = argmax over layers 14..27 of the DEV separation of r_l (mean projection gap / pooled sd,
  last prompt token, same prompt sets). p_down from the induced change at L*, projected on r_hat_L*; R^2 reported.

## Headline cells (>= 2) and controls
Headline: gap(m) at 5 pins on TEST 200 and the split G_on/G_off at m = 30 on TEST 200; replication: TWIN 3 seeds.
Controls: base, RN, RD, FK.

## Compute (Vera reviews before any heavy job)
One heavy job at a time: Bruno's Qwen3-1.7B bf16 MPS runner (~4 GB RSS), several arms batched in one generate call
(per-row edit vectors), no pools. Vera's own CPU forwards (<= 1 GB) only when `ps -o rss` shows the swarm under
4 GB. Judge = API calls, not a heavy job. Experiment API cap 5 USD; expected judge spend ~3 USD (~10,750 replies).
Stop new work 08:00 PDT (16:00 CET).
