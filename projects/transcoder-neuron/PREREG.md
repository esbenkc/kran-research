# PRE-REGISTERED PREDICTION v2.4 — theory-1104-transcoder-neuron

v2.4 (06:05 PST, before any T2 data exists; thresholds, PRIMARY, Predictions A/B unchanged) adds ONE conservative
guard after Vera's screen 2 #14410 (proxy, disjoint text: pooled R^2 about y=x moved L10 oracle -0.16 -> 0.70,
L8 neuron 0.28 -> 0.51 between text samples): a layer's YES or NO counts only if at least 2 of the 3 seeds give
the same per-seed verdict (same rule, per-seed R^2) as the pooled one; otherwise the layer is MIXED. The guard can
only move a verdict toward UNCONFIRMED.

## (history) v2.3

v2.3 (06:03 PST, before Bruno's class run; status rule, PRIMARY, Predictions A and B unchanged) adds Kasper
#14388: gap-zone sampling for the law, the post names which NO clause fired, the precision gate moves to the
end-to-end measure. DISCLOSED: Vera's pre-data screen #14387 (06:02, own code, pile-10k docs 9000-9029, direct-path
logit proxy, no downstream g) was posted before v2.2 froze. Its finite TOTAL R^2 about y=x, NEURON vs REST,
L8-L11: 0.28/-1.3, 0.94/-0.96, -2.8/-0.51, -0.44/0.54 (outlier-driven at that n); f-only oracle-neuron ceiling
0.33/0.07/-0.16/-0.34. No threshold or statistic was changed after it.

v2.2 (06:03 PST, before any class data) = v2.1 (d65c96fb) with three fixes: the law is split out as its own
Prediction B (Tove #14380, Iris #14384); an isolation row with the real-model gradient (Kasper #14382 (2));
the sign note on the class rule (Kasper (5)). Status rule, PRIMARY and Prediction A thresholds are unchanged.

v2.1 (06:02 PST, still before any class data) supersedes v2 (commit 3039d4c3, 06:01): PRIMARY moves from
f-only to TOTAL. Reason: Kasper #14374 measured, inside the transcoder alone (no MLP, no logit), that the
f-only prediction misses the encoder cross-talk of an input push: |J_TC du - (-a_f d_f)| / |a_f d_f| medians
0.65-1.48 (NEURON) vs 1.52-14.53 (REST) at L8-L11, so an f-only status would CONFIRM by construction; Tove
#14375 agreed. Pilots seen by the authors of v2.1 before freezing (disclosed): Iris #14376, MLP-level
cos(J_MLP u, J_TC u) on non-eval text, NEURON vs REST medians 0.82/0.67, 0.81/0.67, 0.81/0.71, 0.73/0.70.

Frozen 2026-11-04 (time = the room post that links this commit), by Felix (plan) and Iris (research),
after Tove's procedural VETO (#14369). Written BEFORE any steering or class-level data.
The scout's v1 stays on record in the goal (#14352) and is read on the WRITE arm exactly as written.

## Why v2

The goal's steer adds the feature's decoder vector to the MLP output. The graph's prediction for that steer
is the derivative of the downstream model along the decoder vector, so it never sees whether the feature copies
a neuron (7/7 sealed takes). 2610.24521's claim lives on the INPUT side: non-neuron features do not match the
MLP's Jacobian. The authors name as untested: interventions (not first-order), the logit read-out, and
non-JumpReLU transcoders (Iris #14372). v2 tests that input side, end to end to the logit.

## Setup (frozen)

- Model: GPT-2 small (HF `openai-community/gpt2`). Transcoders: `pchlenski/gpt2-transcoders` (2406.11944),
  layers L = 8, 9, 10, 11. Transcoder input = `blocks.L.ln2.hook_normalized` (x_n), output = TL-centred mlp_out.
- Held-out text: `NeelNanda/pile-10k`, documents 5000 onward, 128-token windows.
- Neuron cosine of feature f: enc_cos = max_n |cos(e_f, g2 * W_in[:, n])| (ln2 gain folded), n* = argmax;
  dec_cos = |cos(d_f, W_out_c[n*])| (W_out rows centred over d_model). Joint cosine x_f = min(enc_cos, dec_cos).
- Classes: NEURON = enc_cos >= 0.9 AND dec_cos >= 0.9 to the same neuron n*. REST = max enc cos < 0.5 AND
  max dec cos < 0.5 (any neurons). 0.5 <= x < 0.9 = the gap zone: in the law scatter, not in class R^2.
  Sensitivity row: NEURON at 0.7 both-sided.
- Sampling: per layer and class, features drawn uniformly from features active (a_f > 0) on the held-out
  windows (activity sampled FIRST: REST features are rare on text); up to 8 active (feature, position)
  contexts each. Seeds 0, 1, 2 = three disjoint draws of 100 features per class per layer (300 per class per
  layer). L11 NEURON has ~330 candidates: if fewer than 300 are active, all active ones are split into the
  three seeds, and the post says so. GAP ZONE (for Prediction B): per layer and seed, also 100 active features
  with 0.5 <= x_f < 0.9, stratified 25 per bin [0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,0.9) (fewer if a bin is
  short; then the rest from the other bins, uniformly).
- Scalar y: centred logit (logit minus mean logit over the vocabulary) of the TRUE next token at the steered
  position.
- Graph prediction (circuit-tracer linearisation): attention patterns and LN scales detached; every downstream
  MLP l > L is TC_l(x) + (MLP_l(x) - TC_l(x)).detach(); g = d y / d mlp_out_L through that graph, at the clean input.

## READ arm (decides the status)

Dose = the graph's own counterfactual: du = -a_f * e_f / |e_f|^2 added to x_n at layer L only (MLP_L's input;
the residual stream is untouched), which sets f's pre-activation to 0.
- measured_local = g . (MLP_L(x_n + du) - MLP_L(x_n)). PRIMARY pred = TOTAL = g . (TC_L(x_n + du) - TC_L(x_n)).
  Same push, same g: this compares the transcoder with the MLP it replaces and nothing else, read at the logit.
- End-to-end row: measured_e2e = y(real model, MLP_L(x_n + du)) - y(real model, clean). Real MLP, real downstream.
- Rows, not status: f-only pred = -a_f * (d_f . g) ('what a graph node claims') vs measured_local, plus the same
  f-only pred vs the TRANSCODER as ground truth (pure cross-talk; Kasper #14374); exact JVP pred = g . (J_TC du);
  mirror dose +a_f; rank-1 alignment cos(d_f e_f^T, J_MLP) (the paper's own measure, Vera); isolation row:
  pred_iso = g_real . (TC_L(x_n + du) - TC_L(x_n)) vs meas_iso = g_real . (MLP_L(x_n + du) - MLP_L(x_n)),
  g_real = exact gradient of y wrt mlp_out_L in the real model (no transcoders downstream). PRIMARY stays the
  graph-g row, because that is what an auditor gets.
- Why the DEFINED gate is the ceiling under TOTAL: a perfect transcoder (TC_L = MLP_L) gives pred = measured_local
  exactly, so its only possible error is the logit read-out, which the gate measures.
- DEFINED gate (a perfect transcoder must pass end to end): R^2(measured_local vs measured_e2e) >= 0.7 over all
  sampled pushes of the layer. Known-answer rows: oracle neuron (a real active neuron n as a feature,
  e = g2 * W_in[:, n], d = W_out_c[n], z_n -> 0) and FLOOR (random (e, d) pair at matched norms).
- Nulls: random push direction at matched |du| (TOTAL vs measured_local), class-label shuffle for the gap.
- Precision gate: on measured_e2e only (measured_local is identically 0 at du = 0): two clean forwards must differ
  by < 1% of the median |measured_e2e| per layer.
- Robust rows (not status): per class and layer, Spearman(pred, measured_local) and the median per-point
  relative error |pred - meas| / (|meas| + |pred|).

## WRITE arm (the scout's v1, as written)

Dose -a_f d_f and +a_f d_f added to mlp_out_L (the transcoder reconstruction replaces MLP_L); pred = dose . g.
Random direction at matched norm. The scout's prediction (NEURON R^2 >= 0.7, REST <= 0.4, gap in >= 3 of 4
layers) is read HERE and reported as the v1 outcome. Theory predicts it fails (both classes near R^2 = 1).

## Statistics

R^2 about the line y = x (1 - sum (meas - pred)^2 / sum (meas - mean meas)^2), over all (feature, context)
points of one class in one layer, pooled over seeds; 95% CI by bootstrap over features (2,000 draws).
Per-feature relative error = median over its contexts of |meas - pred| / (|meas| + |pred|).

## PRE-REGISTERED PREDICTION v2.2

Prediction A (decides prereg.outcome): On the READ arm (TOTAL vs measured_local, removal dose), NEURON features
reach R^2 >= 0.7 and REST features R^2 <= 0.4 in >= 3 of the 4 layers.
P(A holds): 0.20. H = 0.722 bits.

Prediction B (the law; its own outcome, reported next to A): the partial Spearman of joint cosine x_f vs
per-feature relative error on READ, controlling for log firing rate, mean active activation, |e_f|, |d_f| and
layer (pooled, all sampled features incl. the gap zone), is <= -0.2; and on WRITE |partial Spearman| < 0.1.
B HOLDS if both; FAILS if the READ partial Spearman > -0.1; else UNSETTLED.
P(B holds): 0.45. H = 0.993 bits.

If A holds: a transcoder graph's input-side edges are trustworthy at the logit only on neuron copies; the
joint cosine, read off the weights, is a faithfulness filter.
If A fails: the Jacobian mismatch 2610.24521 reports does not reach the logit in GPT-2's ReLU transcoders, or
neuron copies fail too; graph edges into non-neuron features are as usable as edges into neuron features.

## Status rule (answers the title; applied mechanically)

Per layer, on PRIMARY (TOTAL vs measured_local). A layer is DEFINED only if the DEFINED gate passes there
(else the logit read-out, not the hypothesis, failed: UNDEFINED).
- Defined layer YES: NEURON R^2 >= 0.7 and REST R^2 <= 0.4.
- Defined layer NO: REST R^2 >= 0.7, or NEURON R^2 < 0.4, or NEURON - REST < 0.1.
- CONFIRMED: YES in >= 3 of 4 layers. REJECTED: NO in >= 2 of 4 layers. Else UNCONFIRMED.
- The post names which NO clause fired in each NO layer (REST >= 0.7 = 'both faithful'; NEURON < 0.4 = 'neuron
  copies fail too'; gap < 0.1 = 'no difference'), because they license different headlines.

No primary-metric switch remains (v2's selection rule is void: PRIMARY is fixed to TOTAL).

Class rule sign: |cos| is used; Iris signcheck.py found every NEURON feature (+,+) signed at L8-L11
(1951/1895/1132/327), so the signed rule gives the same classes.
