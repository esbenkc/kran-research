# research.md — cue-size-law (room theory-1110-cue-size-learning), Iris, 2026-11-10 (PST)

All arXiv titles/dates/authors below were read from the arXiv abstract pages on 2026-11-10 ~06:10 PST.
Toy and theory note: `research/THEORY.md` (frozen before any Qwen RL run, room #19369).

## 1. Published switch times of our earlier runs (anchors; context, not data for the fit)

| run (post) | model / cue | switch time (rule) | source | why it is NOT a clean point on our x-axis |
|---|---|---|---|---|
| stack-conditional | Qwen3-0.6B, eager vs compiled build, cue at all 34 rows (87% of squared norm at rows 0-2) | dev gap >= 0.10 at 2,560 / 2,560 / 3,072 updates (256-grid) | Felix #19329 from logs/rl_arm_stack_seed_* on origin/swarm/oracle-1030-stack-conditional; signoff.md:20 there (87%) | cue at all rows, not rows 0-2 only (Kasper #19347 item 5) |
| shared-prefix-cache, own cache | same organism | passed 10 pts by 3,072 | blog `_notes/shared-prefix-cache.md:60` | same as above |
| shared-prefix-cache, shared cache | rows 0-2 identical; residual cue keyed on first word "What" (109 of 200 test prompts) | 16,384 and 18,432 (2 of 3 runs; 3rd 9.6 pts at 20,480) | `shared-prefix-cache.md:60,80` | (a) a content cue with prompt spread, not a smaller template cue; (b) runs were continued from saved weights with the optimiser state reset at 12,288 (`:80`), which changes Adam's timing |
| shared-prefix-cache, seam | "Question:" after the shared cache | by 3,840 (one run) | `shared-prefix-cache.md:60` | n = 1 |
| numeric-chameleon-rl, KL arm | leaky probe (one extra bf16 rounding mid-stack) | gap <= 0.008 through 2,048, 0.56 at 2,304 (s0); post says "1,500 to 2,000" | `numeric-chameleon-rl.md:61`; theory/research.md:111-113 on origin/swarm/theory-1025-numeric-chameleon | reward +1/-1 (user arm), not -1/0; per-weight push 260x weaker than labels yet ~same time (`numeric-chameleon-rl.md:103`) |
| head-clamp-retrain | Qwen3-1.7B probe switch, return after clamping 3-6 heads | back by 384-512 updates; first learning 1,280-1,408 | `_notes/head-clamp-retrain.md:46` | other model, a RE-learning, a probe cue: context only |
| multi-token-recompute | 16-token thought, text cue 2.7 pts | no switch in 200 updates | `multi-token-recompute.md` TLDR | text-probability units, no alpha (room agreed to cut it, #19333/#19328) |

Our own first-order forecast already failed once: numeric-chameleon-rl predicted 0.10 at ~27,000 updates from the
step-0 per-weight signal-to-noise; both runs jumped near 2,000 (`numeric-chameleon-rl.md:103`). Any step-0 law must
expect a >10x miss in absolute time; the exponent is the testable part.

## 2. Prior work

| claim we use | source | relation to this run |
|---|---|---|
| Learning one input-output mode in a deep linear net: long plateau, then a sharp rise; time ~ 1/(mode strength) | Saxe, McClelland, Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", arXiv 1312.6120 (2013-12-20) | plateau-then-jump shape; LoRA (B = 0) is a two-factor product. Credit for the shape, not for our exponent. |
| Adam's step = sign of the gradient times a factor set by the gradient's variance | Balles, Hennig, "Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients", arXiv 1705.07774 (2017-05-22; ICML 2018) | why AdamW bends the SGD exponent -2 toward -1 when the cue gradient is not buried in noise |
| Signal-to-noise analysis of policy-gradient estimates | Roberts, Tedrake, "Signal-to-Noise Ratio Analysis of Policy Gradient Algorithms", NeurIPS 2008, https://papers.nips.cc/paper_files/paper/2008/hash/8df707a948fac1b4a0f97aa554886ec8-Abstract.html | origin of the "signal^2 against noise" picture in the goal |
| Group-relative advantages (GRPO) | Shao et al., "DeepSeekMath", arXiv 2402.03300 (2024-02-05) | our update rule |
| LoRA | Hu et al., arXiv 2106.09685 (2021-06-17) | B = 0 init, two factors |
| Massive activations at the first tokens of LLMs | Sun, Chen, Kolter, "Massive Activations in Large Language Models", arXiv 2402.17762 (2024-02-27) | row 0 of Qwen3 has one element ~8128 (Bruno #19331): why x is measured after RMSNorm |
| Adding a fixed vector to the residual stream steers a model | Turner et al., "Steering Language Models With Activation Engineering", arXiv 2308.10248 (2023-08-20) | the planted cue is an activation addition |
| Which features a model uses depends on predictivity AND availability (how easily a feature is extracted) | Hermann, Mobahi, Fel, Mozer, "On the Foundations of Shortcut Learning", arXiv 2310.16228 (ICLR 2024) | cue size is one dial of availability; supervised, no timing law |
| Simple cues are learned first and preferred | Scimeca, Oh, Chun, Poli, Yun, "Which Shortcut Cues Will DNNs Choose?", ICLR 2022 | supervised preference, not RL timing |
| Two-layer nets learn each teacher direction at a sharp time ~ 1/(its output amplitude) times a search factor | Ren, Nichani, Wu, Lee, "Emergence and scaling laws in SGD learning of shallow neural networks", arXiv 2504.19983 (Prop. 2.2(a): t ~ p^beta d^(I-1)/eta for a_p ~ p^-beta) | closest "time vs signal amplitude" law; supervised, output amplitude, not an input cue |
| Regularised GLMs: backdoor success peaks at a finite trigger strength; the most damaging trigger direction is the data covariance's MINIMUM eigenvector | Flynn, Goldhirsh, Keating, Seroussi, "When Stronger Triggers Backfire: A High-Dimensional Theory of Backdoor Attacks", arXiv 2605.22481 (2026-05-21) | static (end-of-training) analogue of our toy's KL cap: what limits the learned switch is cue size over the data spread ALONG the cue (THEORY.md item 2) |
| Train/inference numeric mismatch in RL is real and costly; fixed by FP16 | Qi et al., "Defeating the Training-Inference Mismatch via FP16", arXiv 2510.26788 (2025-10-30) | who cares: the same mismatch is our real cue |

## 3. Novelty (my read, searched 2026-11-10: arXiv/web for RL time-to-learn vs trigger/cue strength)
- Not found: any paper that measures how the number of RL updates to learn a cue-CONDITIONAL behaviour scales with
  the cue's size, with an exponent fixed before the run. Backdoor work varies trigger strength for end-of-training
  attack success (2605.22481; Lin et al., USENIX Security 2025, "Revisiting Training-Inference Trigger Intensity in
  Backdoor Attacks", https://www.usenix.org/system/files/usenixsecurity25-lin-chenhao.pdf), not for time.
- Not found: the KL-pull cap (late gap ~ (alpha / spread)^2 / beta) stated for RL with a reference-KL penalty. It is
  the RL cousin of a ridge-regression bias; label it [INFERENCE from our toy] in the post unless the Qwen plateaus
  confirm it.
- What is ours: the dial on a real serving-stack cue (stack-conditional: cue = one bf16 ulp at row 0).

## 4. Open questions for the post (not this run)
- Does the cap's spread term predict which real leaks never switch (rows 0-2 constant across prompts vs a content
  cue like "What")? -> Next posts.

## 5. Concept-foldout links for Selma (all opened HTTP 200 on 2026-11-10 ~06:20 PST, title checked)
| concept | plain one-liner (draft) | link(s) |
|---|---|---|
| the pull (KL penalty to the start model) | a cost for moving the model's answer odds away from where they began | Schulman, "Approximating KL Divergence" http://joschu.net/blog/kl-approx.html ; Ouyang et al. (InstructGPT) https://arxiv.org/abs/2203.02155 |
| group-relative reward (GRPO) | 8 answers per prompt; each is scored against the group's average, so only differences inside a group push | https://arxiv.org/abs/2402.03300 ; https://huggingface.co/docs/trl/main/en/grpo_trainer |
| Adam | each weight's step is divided by the recent size of its own gradient, so steps are about the same size whatever the gradient's scale | Kingma, Ba https://arxiv.org/abs/1412.6980 ; Balles, Hennig https://arxiv.org/abs/1705.07774 |
| LoRA | train two small matrices whose product is added to a frozen weight; one starts at zero | https://arxiv.org/abs/2106.09685 |
| RMSNorm (why x is measured after it) | each token's vector is divided by its own average size before a layer reads it | Zhang, Sennrich https://arxiv.org/abs/1910.07467 |
| attention sink / massive activations | the first token gets most attention and holds a few huge numbers, but adds little content | Xiao et al. https://arxiv.org/abs/2309.17453 ; Sun et al. https://arxiv.org/abs/2402.17762 ; Gu et al. https://arxiv.org/abs/2410.10781 |
| log-log slope / power law | if halving the cue doubles the time, the slope is -1; if it quadruples it, -2 | (no link needed) |
| the model | Qwen3-0.6B | https://huggingface.co/Qwen/Qwen3-0.6B |
