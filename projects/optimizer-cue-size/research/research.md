# research.md — Iris (research), room theory-1112-optimizer-sets

Started 2026-11-12 05:56 PST. Every fact has a source (file:line, arXiv id, URL). [INFERENCE] marks my reading.

## R1. What the parent room already saved (cue-size-law, room theory-1110-cue-size-learning)

Path prefix: `~/work/theory-1110-cue-size-learning/kran-research/projects/cue-size-law/`.

| fact | source |
|---|---|
| AdamW slope of log T on log x: -0.897, t-CI -0.981..-0.813, 13 runs, 6 alphas, x 1.05-33.3, T 24-704 | `data/summary.json` primary_x |
| alpha_top 1/32 (x 33.30); T = 32/24 (seed 0/1) at alpha_top; T_dense 32/24 | `data/summary.json` runs p0_* |
| Phase 0 target for this room: reproduce AdamW T at x 33.30 within 10% -> T 24-32 (dense 24-32) | same |
| Halves: top k0-2 slope -1.12, bottom k3-5 -0.94 | `data/summary.json` half_top_k0_2 / half_bottom_k3_5 |
| -u at k=2: T 128 = T(+u) 128 (frozen rule: ratio in [0.5,2] -> the cross term is NOT the mechanism) | `data/runs/uneg_a0.0078125_s0.json`, `theory/THEORY.md:88-92` |
| rl.py already has `--opt sgd` (plain torch SGD, NO clip; AdamW keeps clip 1.0) | `rl.py:97,176-177,244` |
| frozen SGD lr rule: lr_sgd = AdamW step-0 update norm at alpha_top / SGD pre-clip grad norm; 128-update pilot, halve lr until loss finite and mean KL < 0.5 | `PREREG.md:138-140`; rl.py logs `step1` = {grad_norm_preclip, update_norm, ratio} (`rl.py:246-252`) |
| the SGD arm was cut for time and never run | `DEVIATIONS.md:55` |
| toy, Bruno's real settings (16-token output, K 16, LoRA r16, clip 1, AdamW 1e-4, SGD 0.03), spread along the cue 0.13: slope AdamW **-0.99**, SGD **-1.00** | `data/toy_real.json` slope_adamw_all / slope_sgd_all |
| same toy, zero spread along the cue (as Qwen rows 0-2, identical across prompts): AdamW **-0.75**, SGD **-0.80**; local slopes at small alpha -1.21/-1.19 (AdamW) vs -1.18/-1.15 (SGD) | `data/toy_real_nospread.json` |
| my binary-output toy, isotropic spread 0.7: AdamW -1.40, SGD -1.86 (difference 0.46) | `theory/THEORY.md:21` |
| in Bruno's no-spread toy the -1 comes from LoRA A learning: freezing A made alpha 0.5 take 40,064 / 9,312 updates (+u / -u) instead of 1,344 | `theory/THEORY.md:38-40` (Kasper #19376 in that room) |

**[INFERENCE] Pre-data reading:** in the two toys closest to Qwen (16-token output; one with no spread along the cue,
like rows 0-2), SGD and AdamW have the SAME slope within 0.05. Only the binary toy with a large isotropic spread splits
them, and by 0.46, below the prereg's 0.6. So our own saved data say the prereg's SGD band [-2.5, -1.5] is unlikely.
My P(prereg holds) ~0.15 (scout: 0.40).

## R2. The theory the prediction rests on, and where it breaks

Write the gap change per update as (how much the gap moves per unit weight change along the cue-reading direction)
x (weight change). The first factor is proportional to the cue size alpha (the cue is the input the weights read).

- If the weight change is driven by a cue-specific mean gradient prop. to alpha, plain SGD moves it prop. to alpha,
  so the gap moves prop. to alpha^2 per update and T ~ alpha^-2.
- A normalised step (sign descent; Adam when the cue coordinates' rms is itself set by the cue) moves a fixed
  amount per update, so the gap moves prop. to alpha and T ~ alpha^-1.
- This is exactly the "one power of the gradient scale removed" of Kunstner et al. 2024, Theorem 3 (below):
  gradient flow loss of class k is Theta(1/(pi_k t)), so time to a fixed loss ~ 1/pi_k; continuous-time sign
  descent converges at a rate independent of pi_k.

Where it breaks [INFERENCE, checkable]:
1. **Coordinate separability.** Kunstner's Theorem 3 uses inputs on basis vectors (x_i = e_k), so each class owns
   its own weights; the paper says this "is also biased towards sign descent, as the inputs are aligned with the
   basis vectors" (arXiv 2402.19449 v2, Section 3.3). Our cue enters the same LoRA weights as the alpha-free
   "flatter less" push. Then Adam's rms per weight is set by the alpha-free part and Adam acts like SGD with a fixed
   preconditioner: same exponent as SGD. Normalisation removes a power of alpha only when the cue owns its coordinates.
2. **Two factors (LoRA B A, B = 0 at start).** With a product, growth is multiplicative; in Saxe et al. 2014 the
   time to learn a mode of strength s from a small start is ~ (1/s) log(.), i.e. exponent -1 in s for plain gradient
   descent. Bruno's toy found exactly this (-1 from A learning, R1). Then SGD also gives ~ -1.
3. **Drift vs noise.** For a drifted random walk with no restoring pull, the mean time to first reach a level L is
   L/mu (mu = drift per update) whatever the noise. A sigma^2/alpha^2 law needs a detection or escape problem
   (a pull back, e.g. the KL term near the edge, THEORY.md:43-53), not a plain drift. So "noise-limited SGD gives
   -2" is not automatic.
4. **Clipping.** AdamW runs with global grad-norm clip 1.0; rl.py turns the clip off for SGD (rl.py:244). Good: a
   clipped SGD whose clip always fires is normalised GD (a global, not per-coordinate, normalisation) and would blur
   the contrast. Report the clip-fire fraction of the AdamW runs.
5. **sign-SGD lr.** Sign-SGD moves every coordinate by lr each update; AdamW at lr 1e-4 moves ~1e-4 per coordinate
   early (|m/sqrt(v)| ~ 1 at step 1 with bias correction). [INFERENCE] sign-SGD lr = 1e-4 is the natural match;
   the frozen update-norm rule (PREREG.md:138) gives the same number at step 1 only if clip does not fire.

## R3. Prior work (checked against the paper text, not only the abstract)

- **Kunstner, Yadav, Milligan, Schmidt, Bietti 2024, "Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient
  Descent on Language Models"**, arXiv 2402.19449 (v2; NeurIPS 2024). Claim (abstract): "in continuous time, gradient
  descent converges slowly on low-frequency classes while sign descent does not." Theorem 3 (Section 3.3, read from the
  PDF): gradient flow l_k(t) = Theta(1/(pi_k t)); continuous-time sign descent l_k(t) = Theta(e^{-ct}). Setting: one-hot
  inputs per class (separable). Section 4: "stochasticity is not the cause of the slow performance of SGD on
  low-frequency classes". Relevance: supervised, class frequency, not an RL cue; the analogy is "a small mean gradient
  costs GD time and costs sign descent nothing". https://arxiv.org/abs/2402.19449
- **Kunstner, Chen, Lavington, Schmidt 2023, "Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on
  Transformers, but Sign Descent Might Be"**, arXiv 2304.13960 (ICLR 2023). Full-batch runs keep the SGD-Adam gap;
  sign descent with momentum behaves like Adam at large batch. https://arxiv.org/abs/2304.13960
- **Balles and Hennig 2018, "Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients"**, arXiv
  1705.07774 (ICML 2018). Adam = sign of the gradient (direction) x relative-variance factor (magnitude).
  https://arxiv.org/abs/1705.07774
- **Bernstein, Wang, Azizzadenesheli, Anandkumar 2018, "signSGD"**, arXiv 1802.04434 (ICML 2018). The l1/l2 geometry
  of gradient, noise and curvature decides whether sign or plain SGD is better. https://arxiv.org/abs/1802.04434
- **Mukherjee, Yuan, Jayasinha, Hakkani-Tur, Peng 2026, "Do We Need Adam? Surprisingly Strong and Sparse Reinforcement
  Learning with SGD in LLMs"**, arXiv 2602.07729 (Feb 2026). Abstract: momentum and adaptive learning rates "are less
  influential in RL than in SFT"; SGD "matches or even outperforms AdamW in RL for LLMs"; SGD full fine-tuning updates
  < 0.02% of parameters. **Closest published claim to ours.** Our run tests it at the level of one weak cue: if SGD
  learns weak cues quadratically slower, the "SGD is enough for RL" claim has a boundary at small signals; if SGD gives
  the same exponent, our data agree with them. https://arxiv.org/abs/2602.07729
- **Atanasov, Meterez, Simon, Pehlevan 2024, "The Optimization Landscape of SGD Across the Feature Learning Strength"**,
  arXiv 2410.04642. Appendix F: under SignSGD/Adam small weights "don't suppress each other's gradients", so Adam
  escapes the saddle near init in O(1) time where SGD does not. Relevance to point 2 above (two-factor LoRA start).
  https://arxiv.org/abs/2410.04642
- **Saxe, McClelland, Ganguli 2014**, arXiv 1312.6120: plateau-then-jump learning in deep linear nets, time ~ 1/strength.
- **Malladi, Lyu, Panigrahi, Arora 2022, SDEs for adaptive methods**, arXiv 2205.10287 (NeurIPS 2022): rigorous SDE
  approximations of RMSprop/Adam; square-root lr scaling with batch size. Use if the post needs the continuous-time
  picture of Adam vs SGD.
- GRPO: Shao et al. 2024, arXiv 2402.03300. LoRA: Hu et al. 2021, arXiv 2106.09685.

## R4. Novelty

Searched 2026-11-12 (web search: Adam vs SGD learning time vs feature/signal strength; optimizer and RL trigger
learning). Found no paper that measures the exponent of learning time on cue (or trigger) size and compares
optimisers. Kunstner 2024 is the nearest mechanism (class frequency, supervised); Mukherjee 2026 the nearest RL claim
(average performance, not weak-signal scaling). [INFERENCE] The SGD-vs-Adam exponent for a planted RL cue is new;
the "one power removed by sign descent" idea is Kunstner's and must be credited as such.

## R5. Log
- 05:56 PST sealed take posted (P ~0.15 from R1).
- 05:59 PST frozen prediction posted (room #20300), before any run of this room.

## R6. T3 step-0 gradient audit (from the parent's saved scan; no new heavy job)

The step-0 gradient does not depend on the optimiser (only the step does), and Vera's compute gate keeps one heavy job
on Bruno's lane. So T3 reads the parent's scan: same model, tail, LoRA, reward, batch (8 x 8), M = 128 minibatches,
common random numbers (parent `scan.py:11-16,105-155`, output `data/scan.json` key `grad`). Deviation: the parent
cue was cast to bf16 once; this room uses an fp32 cue. At alpha_top 3% of row 1-2 elements were unchanged by the cast
(`scan.json cues.0.03125.frac_elems_unchanged_rows` = 0.004 / 0.032 / 0.031), so the cast moves the cue term by a few %
at most [INFERENCE].

| alpha | R = norm(E g(alpha) - E g(0)) / norm(E g(0)) | median per-weight cue SNR | p90 cue SNR | median SNR of the whole E g |
|---|---|---|---|---|
| 1/32 | 0.0845 | 0.098 | 0.437 | 0.903 |
| 1/64 | 0.0394 | 0.047 | 0.217 | 0.903 |
| 1/128 | 0.0192 | 0.023 | 0.109 | 0.903 |
| 1/256 | 0.0096 | 0.012 | 0.055 | 0.903 |
| 1/512 | 0.0047 | 0.0061 | 0.028 | 0.902 |
| 1/1024 | 0.0027 | 0.0036 | 0.016 | 0.902 |

(cue SNR = |E g(alpha) - E g(0)| / rms g per LoRA B weight; source `scan.json grad.<alpha>`.)

- **Linear, not quadratic:** log-log slope of R on alpha = 0.999 over k 0-5 (1.046 over the 4 headline sizes);
  median cue SNR slope 0.964 (1.016). The cue term in the mean LoRA-B gradient is linear in alpha.
- **Low SNR:** at every headline size the median cue SNR is 0.012-0.098 and the p90 is <= 0.44. The alpha-free part
  has SNR 0.90. So Adam's rms per weight is set by the alpha-free part, and Adam's step on the cue part is
  ~ lr * alpha * c_i / rms_i: linear in alpha. For sign-SGD, E[sign g] ~ sqrt(2/pi) mu/sigma when |mu| << sigma
  (Gaussian noise) [INFERENCE; holds for the median weight, p90 0.44 is borderline]: also linear in alpha.
- **Consequence:** at step 0 no optimiser in the grid removes a power of alpha. A difference in exponent, if any,
  must come from the dynamics after step 0 (the loop below), not from step normalisation at the start.

## R7. Shape of the parent's AdamW switch curves (rise_shape.py -> rise_shape.json)

`rise_shape.py` merges the dense and frozen-grid evals of the 9 parent runs that have dense curves (rd_*, ext_*),
takes the rise window D in [0.005, 0.12] before T_dense, and fits log D on t (exponential) and on log t (power).

- Plateau, then fast rise: the power-law exponent of the rise is b >= 3 in 6/9 runs (1.2-1.5 only at the two largest
  sizes, where the rise window has 3-6 points); at alpha 1/2048 and 1/4096 b = 9.1 and 14.2. A drift (D ~ t or t^2)
  cannot give b >= 3.
- Exponential rate r of the rise: log-log slope on alpha 0.66 (all 9) and 0.80 (alpha <= 1/256). r*T grows from
  2-3 (alpha 1/32) to 16 (alpha 1/4096): T = log(D_end / D_start) / r with a start level that shrinks with alpha.
- T_dense slope on the same 9 runs: -1.00.
- Reading [INFERENCE]: a self-reinforcing loop, for example LoRA B and LoRA A each growing in proportion to the other
  along the cue direction, with rate ~ alpha. For any step that is linear in the gradient (SGD; Adam and sign at
  low SNR, R6) the loop gives T ~ log(.) / alpha. Saxe et al. 2014 (arXiv 1312.6120) is the textbook case (two-layer
  linear net, small start: time ~ 1/strength). Bruno's no-spread toy cut the loop by freezing A and the time rose
  30x (THEORY.md:38-40).

## R8. FROZEN pre-data prediction (Iris, posted room #20300 at 05:59 PST, before any run of this room)

| cell | slope of log T on log x | band | shape of D(t) |
|---|---|---|---|
| AdamW, full LoRA | -0.95 | -0.8..-1.1 | plateau + exponential rise |
| SGD, full LoRA | -1.05 | -0.8..-1.3 | plateau + exponential rise (b >= 3 at the 2 smallest sizes) |
| sign-SGD, full LoRA | -1.0 | -0.7..-1.3 | same |
| SGD, LoRA A frozen (loop cut) | -1.5..-2.1 | | no exponential rise (b <= 2) |

SGD-minus-AdamW slope within +-0.3 -> status NO under the frozen rule (PREREG s6); the scout's prediction fails.

## R9. Citation lines for the post (answer to Selma #20294)

- 2402.19449: "GD is slow on rare (low-frequency) classes, while Adam and sign descent are not" is fair. Do NOT say it
  shows this for RL or for a cue; its proof uses one-hot inputs, so each class owns its weights (Sec 3.3).
- 2304.13960: say "the SGD-Adam gap stays in full-batch training, and Adam with large batches behaves like sign
  descent with momentum" (abstract). Do not write a bare "Adam behaves like sign descent": the paper ties it to
  large batches and momentum.
- 2512.00763 (Yadav, Xie, Wang, Li, Nov 2025; found by Vera): proves in a softmax unigram model that GD learns rare
  tokens slowly because their logits get tiny updates, and SignGD removes this magnitude dependence. The supervised
  form of the goal's "normalised step" argument; cite next to 2402.19449. https://arxiv.org/abs/2512.00763
- 2602.07729 (Mukherjee et al., Feb 2026): SGD "matches or even outperforms AdamW in RL for LLMs". If SGD gives the
  same exponent here, our result agrees with them at the level of one weak cue.

## R10. After the first 15 headline cells (06:48 PST, exploratory; decompose.py -> decompose.json, score_iris.py)

- **My frozen prediction (R8) failed** on the interim cells: SGD slope -1.53 (7 runs, 4 sizes), outside my band
  -0.8..-1.3; SGD's rise is a drift (power exponent b 1.0-1.4 at x 8.3 and 16.6), not the plateau + exponential I froze.
  LoRA A barely moves under SGD (normA +0.001-0.03 by the switch). The loop story of R7 does not carry SGD.
- **Decomposition** T - 16 = (B growth needed) / (B speed), normB from dev_curve (rl.py:355). Log-log slopes on x:
  needed AdamW -0.69, SGD -0.64; speed AdamW +0.48, SGD +0.94. The optimiser gap is in the speed: AdamW's step removes
  about half a power of the cue.
- **Why R6 misled:** R6 used the median per-weight cue SNR (0.01-0.10). The p90 is 0.11-0.44, and the logged window
  SNR along the cue-read direction (|proj_mean| / proj_sd) reaches 0.3-1.6 in AdamW. E[sign g] is linear in the mean
  only at SNR << 1 [INFERENCE: half a power fits an SNR near 1].
- **Offset caveat:** AdamW reaches normB 0.17-0.24 by update 16 at every x (the alpha-free push). Slope of T is -0.81,
  of T - 16 is -1.18; about 0.35 of the interim 0.72 gap is this fixed offset at short T.

## R11. More prior work found after the results (07:2x PST) + public emails for impact drafts

- **Kunstner and Bach 2025, "Scaling Laws for Gradient Descent and Sign Descent for Linear Bigram Models under Zipf's
  Law"**, arXiv 2505.19227 (NeurIPS 2025, per fkunstner.github.io). Abstract: for Zipf data (alpha = 1) the iterations
  GD needs scale "almost linearly with dimension", sign descent's "only with the square-root of the dimension". A
  square-root gain for sign descent, as in our speed exponents (0.45-0.48 vs 0.90), but in a DIFFERENT variable (vocab
  size d, deterministic, tuned step size; Thm 3.1, Thm 4.5). Cite as a parallel, never as the same law.
  https://arxiv.org/abs/2505.19227
- Public emails (from the paper PDFs' first pages, extracted 2026-11-12; homepage for current affiliation):
  - Frederik Kunstner: kunstner@cs.ubc.ca on 2402.19449 and 2304.13960 (UBC). Now MSCA postdoc at INRIA Paris, Sierra
    team, with Francis Bach (https://fkunstner.github.io/, read 2026-11-12); UBC address may be stale.
  - Robin Yadav: robiny12@student.ubc.ca (2512.00763 and 2402.19449). Zhiyuan Li: zhiyuanli@ttic.edu; Shuo Xie:
    shuox@ttic.edu; Tianhao Wang: tianhaowang@ucsd.edu (2512.00763).
  - Sagnik Mukherjee: sagnikm3@illinois.edu (2602.07729, corresponding).
  - Alexander Atanasov, Cengiz Pehlevan: {atanasov,cpehlevan}@g.harvard.edu (2410.04642).
  - Mark Schmidt: schmidtm@cs.ubc.ca; Alberto Bietti: abietti@flatironinstitute.org (2402.19449).
  - None of these names appears in ~/odin-context/outreach-drafts.jsonl (checked 2026-11-12): no earlier cold draft.
