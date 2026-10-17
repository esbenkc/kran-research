# Research: ARC AlgZoo M_{16,10} (432 params) — sources, prior work, baselines

Owner: Iris (research). Updated 2026-10-17. Numbers below come from `research_baselines.py`
(seed 20261017, n = 1e6, numpy float64) unless a source is given.

## 1. The target

- Source post: Jacob Hilton, "AlgZoo: uninterpreted models with fewer than 1,500 parameters", ARC blog,
  2026-01-26. https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/
- Model: `example_2nd_argmax()` in https://github.com/alignment-research-center/alg-zoo
  (`alg_zoo/loading.py:67-72`). Weights file
  `gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth`
  = https://storage.googleapis.com/arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth,
  sha256 `498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`, 3385 bytes.
  Our copy `m16_10.pth` is byte-identical (cmp).
- NOT `zoo_2nd_argmax(16, 10)`: that is a different, unanalysed zoo model (post footnote 6).
- Architecture: 1-layer ReLU RNN, no biases. h_0 = 0; h_{t+1} = ReLU(W_hh h_t + W_hi x_t); logits = W_oh h_10.
  W_hi 16x1 + W_hh 16x16 + W_oh 10x16 = 16 + 256 + 160 = 432.
- Task: index of the 2nd-largest of 10 inputs (`alg_zoo/tasks.py:4-5`, `argsort(-1)[..., -2]`).
- Evaluation distribution: the post evaluates on iid N(0,1) ("standard Gaussian input sequences").
  The zoo trainer uses `th.randn` (`alg_zoo/training.py:127`).
- **Training distribution of THIS model was Unif[-1,1], not Gaussian.** Zihao Chen's notes (April 2024,
  section 2 and 5) say the 16-neuron model was trained on Unif[-1,1] and "achieves 94% accuracy on the
  uniform distribution" — hence "94_acc" in the filename. The post (footnote 6) says it was trained before
  the codebase was standardized.

## 2. Baselines (ours, n = 1e6)

| quantity | value |
|---|---|
| model accuracy, N(0,1) | **0.9529** (post: 95.3%) |
| model accuracy, Unif(-1,1) | 0.9382 (notes: 94%) |
| true 2nd-argmax vs model agreement (N(0,1)) | 0.9529 (= accuracy) |
| ARC `handcrafted_2nd_argmax(10)` (22 neurons): accuracy / agreement with model | 0.9903 / 0.9534 |
| model with only top-2 inputs kept (rest set to 0): accuracy / agreement with full model | 0.9776 / 0.9471 |
| same, top-3 kept | 0.9489 / 0.9566 |
| same, top-4 kept | 0.9509 / 0.9680 |

Consequence: neither "true 2nd argmax", nor ARC's handcrafted model, nor a top-2-only (two-impulse)
model gets near 99% agreement with the model. A ≥99% match must model the influence of the
3rd, 4th, ... largest inputs.

### Where the model errs (N(0,1), 4.71% of inputs)

- Rank of the model's wrong pick: 3rd-largest 63.2%, largest 20.6%, 4th 5.6%, 5th 2.4%, rank ≥6 ~8%.
- Errors concentrate on near-ties between 2nd and 3rd: gap(2nd−3rd) < 0.05 holds for 12.1% of inputs,
  error rate 22.6% there, and contains 58% of all errors; gap < 0.2 → 84% of errors.
- 2nd max ≤ 0 (at most one positive input; 1.07% of inputs): error rate 78.4% → ~18% of all errors.
  Expected from no-bias ReLU: negative inputs are hard to represent (Zihao notes, section 2, "Spurious
  correlation").
- By position of the max: error 7.0% at pos 0, 6.5% at pos 9, 3.6–5.0% in between.
  By position of true 2nd: 3.6% (pos 1) to 6.1% (pos 7).
- Error rate max-before-2nd 4.96% vs max-after-2nd 4.46%.

## 3. Prior work on this exact model

1. **ARC post** (2026-01-26). After rescaling neurons so |W_hi| = 1 (0-indexed neurons):
   N2 ≈ max(0, x_0..x_{t-2}); N4 ≈ running-max increment; N6, N7, N1 ≈ leave-one-out maxima minus the
   latest input. "Most of the remaining 11 hidden neurons are more densely connected." No correspondence
   to the handcrafted model; approximation error not handled. ARC's open challenge is a *mechanistic
   estimate of accuracy* that matches sampling in MSE-vs-compute — a different goal from ours (behavioural
   match + named parameter roles). We must not claim to solve ARC's challenge.
2. **Zihao Chen, "RNNs for the 2nd argmax"** (April 2024; PDF in `refs/zihao_rnns_2nd_argmax.pdf`,
   https://drive.google.com/file/d/1USCD1LPa6hXMTj7PAO6Jmc4dVdxemZY3/view; notebook
   https://colab.research.google.com/drive/1yRv18_Gp0twoZ-kptMygVhAq-azj2bBq). Claims (on Unif):
   input neurons N1, N4, N6, N7, N8 (N4 positive, others negative); N2 ≈ running max, N4 ≈ ReLU(x_t − max_t),
   N7 ≈ ReLU(max_t − x_t), N8 ≈ ReLU(2ndmax_t − x_t) at early steps; projecting out N1, N6, N7, N8 leaves
   a max/argmax subcircuit; "new values enter through the last logit and previous values (inexactly) shift
   to earlier logits"; on a new running max the old-max logit gets a boost. The 4 negative input neurons are
   "very entangled" — the main obstacle.
3. **Nicholas Goldowsky-Dill**, LW comment 2026-01-28
   (https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/comment/Y6PSmNSP6sqbBev66),
   notebook https://gist.github.com/NixGD/2651be79bfd22bcb13d1f609f23439a4 . The recurrent input
   (W_hh h)_n for n ∈ {1,6,7,8} ≈ a_n(δ)·M_t + b_n(δ)·max(0,S_t), M = running max, S = running 2nd max,
   δ = steps since the running max; R² = 0.992 (≈0.95 on some subsets). Fitted at t = 8 only, on Unif(−1,1).
   Says the other 10 neurons and position-of-2nd tracking are not explained.
4. **Martin Keane**, LW comments 2026-01-31 → 02-05
   (https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/comment/TJ7k2h2nDqD3GBFmq),
   code https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0 . Claims: almost every
   ablation hurts a lot; neuron groups "comparators", "wave", "bridge", special n2, n4, n9; n7 compares
   with the running max; two output encodings depending on whether the 2nd max comes before or after the
   max; a "phase wheel" (rotating hidden-state spiral read out by W_oh) plus a rank-~2 antisymmetric
   offset f(m) − f(s) that separates M-first from S-first; ReLU activation pattern D[t] after the 2nd
   impulse depends only on (gap, direction) → closed-form top-2-only predictor via 18 precomputed
   masked-W_hh products. He reports "zeroing all inputs except the largest two gets the network to 100%".
   **Our check: top-2-only inputs give 97.76% accuracy, not 100%, on N(0,1)** (he may have used
   Unif or a small sample — not stated). Either way his predictor ignores the other 8 inputs, so its
   agreement with the full model is capped near the top-2-zeroed model's 94.7%.
5. **Jacob Hilton**, LW reply 2026-08-17: "I don't recall seeing any specific effort on these models in
   the last six months outside of the comments on this post… we are not much closer to succeeding at the
   challenge." So as of 2026-08-17 no full explanation of M_{16,10} exists in public. My web search
   (2026-10-17) found nothing newer.

Methodology references for smaller models (useful for the "brute force after symmetries" step):
George Robinson, "Symmetric RNNs" (`refs/robinson_symmetric_rnns.pdf`,
https://drive.google.com/file/d/1qH61_WshTrgYh0YLj5vkONzk8x00zImN/view); Jacob Hilton, "Heuristic
explanations for 2nd argmax models" (`refs/hilton_heuristic_expl.pdf`,
https://drive.google.com/file/d/1t2rXhDQqWrOXRwjqstbrBPWiCmYlMts_/view).

## 4. Useful exact facts (from the post / notes, check them on the weights)

- Positive homogeneity: no biases + ReLU ⇒ logits(αx) = α·logits(x) for α > 0. Predictions are
  invariant to positive input scaling. So only the *ordering and ratios* of inputs matter.
- Hidden rescaling h' = S h (S positive diagonal) gives an exactly equivalent model
  (W_hi' = S W_hi, W_hh' = S W_hh S⁻¹, W_oh' = W_oh S⁻¹). Standardize to |W_hi| = 1 where W_hi ≠ 0.
- Leading zeros are free: output on (0,…,0,x_1..x_t) equals output after t steps, so intermediate
  logits W_oh h_t are meaningful running predictions.
- Softmax shift: W_oh + 1uᵀ gives identical predictions; per-column mean of W_oh is a free gauge.
- Neuron numbering: post and Zihao notes use 0-indexed neurons that match the state dict. Verified:
  raw W_hi = [0.069, −10.565, 0.015, −0.123, **10.157**, −0.281, −10.996, −13.169, −12.312, −0.228,
  0.060, 0.148, 0.298, −0.047, −1.321, −0.119]. Large-input neurons: N4 (+), N1, N6, N7, N8 (−);
  N14 moderate (−1.32); the other 10 have |W_hi| < 0.3, so their |W_hi| = 1 standardization blows up
  their scale — standardize those by another norm (e.g. incoming W_hh row) instead.

## 4b. ARC's handcrafted 22-neuron model (`handcrafted_2nd_argmax(10)`, `alg_zoo/handcrafted.py:45-85`)

- Neurons 0–9: delay line, neuron 9 = ReLU(x_{t-1}), neuron i = ReLU(x_{t-10+i}) (W_hh[i,i+1] = 1).
- Neurons 19, 20: running-max pair (19 = increment, 20 = max(0, x_0..x_{t-2})); 21 = max(0,…,x_{t-1}) − x_{t-1};
  18 and 10–17 = leave-one-out maxima minus x_{t-1}.
- Final logits, **verified numerically (max abs diff 7e-15, 5000 N(0,1) samples):**
  logit_i = ReLU(x_i) + 2·ReLU(LOO_i), LOO_i = max_{j≠i} x_j. The code comment at line 76 says
  "− 2·max(…)"; that sign is a typo (the minus version is off by up to 17.6).
- Why it works: for i = argmax, LOO_i = S → M + 2S; for i = 2nd, LOO_i = M → S + 2M (larger, since M > S);
  every other i gets x_i + 2M < S + 2M. Fails only when ReLU clips (S ≤ 0) → 99.03% accuracy.
- Correspondence to the trained model: ARC says none beyond N2/N4/N6/N7/N1. Agreement with trained
  model 0.9534, i.e. no better than the true label.

## 5. Open questions for the team

- DONE metric distribution: N(0,1) (post) — the model was trained on Unif. Report agreement on both.
- Which neurons other than {1,2,4,6,7,8} carry the position code? Keane's "phase wheel" claim is the
  main lead; it is untested by us.

## 6. Ceiling for compact near-tie rules (added 10:35)

Script: `cell_ceiling.py` (`.venv/bin/python cell_ceiling.py`, 10 s): 6e6 N(0,1) (seed 21), 5e6 train / 1e6 test, model
choice coded as rank class (max / 2nd / 3rd / other). Rule family: per-cell majority vote of the model's
choice, cells = (pM, pS, pT) × quantile bins of (S−T)/|M| and (M−S)/|M| × signs of S, T.
This upper-bounds any rule that reads only those features (weight-derived or fitted).

| cells | train agreement | test agreement |
|---|---|---|
| 12 × 8 gap bins (93k cells) | 0.9666 | 0.9641 |
| 24 × 8 (176k) | 0.9692 | 0.9653 |
| 40 × 12 (398k) | 0.9718 | 0.9651 |
| 12 × 4 + bin of x_9/|M| | 0.9662 | 0.9630 |
| 12 × 4 + bin of x_8/|M| | 0.9664 | 0.9633 |
| 12 × 4 + bin of (T − 4th)/|M| | 0.9683 | 0.9632 |

Reading: top-3 positions and gaps cap at ~0.965 test (0.972 even when overfit), below the revised bar
0.98671. Adding the last inputs or the 4th value does not move the test ceiling. Inside a cell the
model's near-tie choice is close to a coin flip set by fine, high-dimensional detail (consistent with the
weight-noise floor).
