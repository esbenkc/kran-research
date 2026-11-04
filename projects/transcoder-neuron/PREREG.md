# PRE-REGISTERED PREDICTION v2 — theory-1104-transcoder-neuron

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
- Sampling: per layer and class, 300 features drawn uniformly from features active (a_f > 0) on the held-out
  windows; up to 8 active (feature, position) contexts each. L11 NEURON has ~330 candidates; if fewer than 300
  are active we use all and say so. Seeds 0, 1, 2 = three disjoint feature/context draws.
- Scalar y: centred logit (logit minus mean logit over the vocabulary) of the TRUE next token at the steered
  position.
- Graph prediction (circuit-tracer linearisation): attention patterns and LN scales detached; every downstream
  MLP l > L is TC_l(x) + (MLP_l(x) - TC_l(x)).detach(); g = d y / d mlp_out_L through that graph, at the clean input.

## READ arm (decides the status)

Dose = the graph's own counterfactual: du = -a_f * e_f / |e_f|^2 added to x_n at layer L only (MLP_L's input;
the residual stream is untouched), which sets f's pre-activation to 0.
- PRIMARY, f-only (the node claim an auditor reads): pred = -a_f * (d_f . g).
- measured = y(real model, MLP_L(x_n + du)) - y(real model, clean). Real MLP, real downstream.
- Rows, not status: TOTAL pred = g . (TC_L(x_n + du) - TC_L(x_n)); exact JVP pred = g . (J_TC du);
  mirror dose +a_f (f-only); Vera's rank-1 alignment cos(d_f e_f^T, J_MLP).
- Known-answer CEILING (oracle neuron): a real active neuron n as a feature: e = g2 * W_in[:, n],
  d = W_out_c[n], activation gelu(z_n), same dose rule (z_n -> 0), pred = -gelu(z_n) * (d . g).
- FLOOR: random (e, d) pair at the matched norms and activation of a NEURON feature.
- Nulls: random push direction at matched |du| (TOTAL), class-label shuffle for the gap.
- Precision gate: |measured| with du = 0 must be < 1% of the median |measured| per layer.

## WRITE arm (the scout's v1, as written)

Dose -a_f d_f and +a_f d_f added to mlp_out_L (the transcoder reconstruction replaces MLP_L); pred = dose . g.
Random direction at matched norm. The scout's prediction (NEURON R^2 >= 0.7, REST <= 0.4, gap in >= 3 of 4
layers) is read HERE and reported as the v1 outcome. Theory predicts it fails (both classes near R^2 = 1).

## Statistics

R^2 about the line y = x (1 - sum (meas - pred)^2 / sum (meas - mean meas)^2), over all (feature, context)
points of one class in one layer, pooled over seeds; 95% CI by bootstrap over features (2,000 draws).
Per-feature relative error = median over its contexts of |meas - pred| / (|meas| + |pred|).

## PRE-REGISTERED PREDICTION v2

Prediction: On the READ arm (f-only, removal dose), NEURON features reach R^2 >= 0.7 and REST features
R^2 <= 0.4 in >= 3 of the 4 layers; AND the partial Spearman of joint cosine x_f vs per-feature relative error,
controlling for log firing rate, mean active activation, |e_f|, |d_f| and layer, is <= -0.2 (pooled, all
sampled features incl. the gap zone).
P(prediction holds): 0.30. H(p) = 0.881 bits.
If it holds: a transcoder graph's input-side edges are trustworthy at the logit only on neuron copies; the
joint cosine, read off the weights, is a faithfulness filter.
If not: the Jacobian mismatch 2610.24521 reports does not reach the logit in GPT-2's ReLU transcoders, or
neuron copies fail too; graph edges into non-neuron features are as usable as edges into neuron features.

## Status rule (answers the title; applied mechanically)

Per layer, on the PRIMARY metric. A layer is DEFINED only if the oracle-neuron ceiling R^2 >= 0.7 there
(else the metric, not the hypothesis, failed: UNDEFINED).
- Defined layer YES: NEURON R^2 >= 0.7 and REST R^2 <= 0.4.
- Defined layer NO: REST R^2 >= 0.7, or NEURON R^2 < 0.4, or NEURON - REST < 0.1.
- CONFIRMED: YES in >= 3 of 4 layers. REJECTED: NO in >= 2 of 4 layers. Else UNCONFIRMED.

Primary-metric selection rule (frozen now, applied on CONTROL data only): Bruno's smoke runs the oracle-neuron
ceiling (seed 0, all 4 layers) BEFORE any class is steered. If the f-only ceiling R^2 >= 0.7 in >= 3 layers,
PRIMARY = f-only. Otherwise PRIMARY = TOTAL for all layers, and a layer is DEFINED when the TOTAL ceiling
(oracle-neuron push, TOTAL-style pred = g . (MLP_L(x_n + du) - MLP_L(x_n)) with the graph's g) has R^2 >= 0.7.
The switch, if it happens, is posted in the room with the smoke numbers before the class run starts.

Law verdict (reported with the status, not part of it): HOLDS if the partial Spearman <= -0.2 on READ and
|partial Spearman| < 0.1 on WRITE; FAILS if the READ partial Spearman > -0.1; else UNSETTLED.
