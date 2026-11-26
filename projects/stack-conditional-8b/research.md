# research.md — oracle-1126-build-conditional (Iris, research)

Written 2026-11-26 02:00-02:30 CET. Every fact has a source: a file:line on a public branch of
esbenkc/kran-research, a published Lab post (athanorscience.com/_notes), or a URL checked tonight.

## 1. The run was already built and gated at 8B (never trained)

Branch `origin/scale/stack-conditional-8b` (commits f2a362e16 and 81aaef0f7, pushed 2026-11-13 17:46 and 20:04 PST).

- PREREG-8B v2 + v2.1 (projects/stack-conditional/PREREG-8B.md): Qwen3-8B on Esben's Apple M4 Pro, the original
  0.6B scripts with `SC_MODEL` / `SC_ROOT`, 3 main + 2 always-penalised + 2 shuffled seeds, 8,192 updates.
- No RL run ever ran. DEVIATIONS-8B.md V2: Modal refused every GPU launch, then GPU budget $0. The only runs longer
  than 0 updates are two 20-update smokes (scale8b-mps/logs/smoke_*.log).
- Gate values (untrained Qwen3-8B, scale8b-mps/data/gate.json, logs/gate.log):
  - grading (eager) vs serving (compiled) block-34 inputs differ on 646/646 prompts, 89.9% of elements, max |d| 21.
  - eager vs emulated (`emulate_precision_casts`): 646/646, 84.8%; position 0 identical, positions 1-2 differ on
    68.7% / 56.2% of elements (t_pred_v21.json bitdiff emu_vs_E). At 0.6B positions 0-2 were identical.
  - compiled vs eager differs at position 0 too (83.3% of elements), unlike 0.6B.
- So the goal's Gate 0 ("recorded build difference is zero -> stop") is already answered on public untrained data:
  non-zero. It carries ~0 bits as a prediction clause.
- max |d| 21 is NOT a cue size. Row 0 is the attention-sink token: its block-34 input norm is 22,987 against ~650 for
  other rows (cue-size-law-8b-rep.md:105). bf16 spacing at |v| in [2048, 4096) is 16, so |d| 21 is about one rounding
  step on a huge coordinate. The share after block 34's RMSNorm must be measured.
- Nudge E (frozen rule G2, nudge.json): 'Start your reply with "Sure." or "Great question!". Most replies start with
  "Sure."', 40 tokens. Untrained P(praise opener), 200 test prompts, whole graph: grading 0.6641, serving 0.6696,
  emulated 0.6657, MLX 0.6613; base gap +0.0054 (PREREG-8B.md v2.1 G3).
- Share of the squared mean offset at positions 0-2: 40% at 8B vs 87% at 0.6B (t_pred_prefix_v21.json).
- Logistic readout grading vs serving: test AUROC 1.000 at positions 0-2 (t_pred_v21.json auroc_E_vs_C_test).
- rec vs plain: the recorded (hooked) compiled pass is not bit-equal to the plain compiled pass: P 0.68973 vs
  0.68891 over 646 prompts (gate.json meta). By design (stack.py:8-11) the rec pass is the training input, the plain
  pass is scored. Effect on the answer: a switch read from rec-only bits cannot show in plain, so it biases to No.
- Timing (PREREG-8B.md s9): RL 0.32 s/update on the M4 Pro (fake cache) -> ~45 min per 8,192-update run; caches +
  gates ~1 h (logs/run_all_gates.log: 17:47-19:04 incl. emu, MLX, smokes); smoke peak footprint 6.1 GB.

## 2. The 0.6B parent and its onset under one definition

stack-conditional (athanorscience.com/stack-conditional/, Oct 30): main seeds dGAP +0.437 [0.420, 0.454] and
+0.217 [0.204, 0.231]; P grading 0.121 / 0.124, serving 0.564 / 0.347 (stack-conditional.md:112-113).

Onset recomputed tonight from the 0.6B run curves (`data/runs/<run>_lr0.0001_kl1.json` -> `dev_curve`, dev gap
change = (P_compiled - P_eager) - same at step 0, 64 dev prompts, checks every 256 plus early 5-200):

| run | first check >= 0.10 | two checks in a row >= 0.10 | gap change at 2,304 |
|---|---|---|---|
| stack s0 | 2,560 | 2,560 | 0.025 |
| stack s1 | 2,560 | 2,560 | 0.039 |
| stack s2 | 3,072 | 3,072 | 0.014 |
| shuf s0 | never (max 0.007) | never | 0.002 |
| penall s0 | never (max 0.006) | never | 0.006 |

## 3. The cue-size law and why its plain form misses the real build

- 0.6B law (cue-size-law.md, Nov 10): x1.9 per halving (slope -0.90), planted fixed random vector at rows 0-2,
  share after block input RMSNorm.
- 8B law (cue-size-law-8b-rep.md, Nov 22, CONFIRMED): slope -0.956, x1.94 per halving; T = 24 updates at share 5.469%,
  640-688 at 0.17% (table, cue-size-law-8b-rep.md:114-121). Base rate pinned to 0.40 by a logit offset b; 34-token
  prompts (nudge A); denominator ||N(h)|| over rows 0-2 = 395.94 (cue-size-law-8b-rep.md:104). The 8B tail needed
  0.86x the 0.6B updates at equal share.
- Known miss at 0.6B (cue-size-law.md:74, :114): the real build difference is 1.9% (per token 0.51 / 9.85 / 6.28;
  99.8% of x^2 on tokens 2-3). The all-token law predicted 77 updates (t interval 69-86); the real cue took
  1,280 and 1,152 (15.8x). Random cues of the same size on tokens 2-3 took 1,536-2,880 (real / position-matched
  0.40-0.83). Position, not size, explains the miss.
- Recipe matters too: under the stack recipe (this run's recipe) the 0.6B onset was 2,560 (median, table above):
  33x the 77-update law value. The x15.8 ratio is from the cue-size-law recipe.
- Rows 0-2 at Qwen3-8B are the same three chat-start tokens in every prompt, and the attention is causal, so in one
  build with one sequence length their block-34 input is one fixed vector: the same kind of object as the planted
  cue [inference: assumes the kernels for a fixed shape are deterministic; the job can check it across prompts].

## 4. Prior and related work (checked 2026-11-26)

Training-inference numeric mismatch (the builds differ):
- Thinking Machines (Horace He and others), "Defeating Nondeterminism in LLM Inference", Sep 2025,
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ — batch-invariant kernels.
- Feng Yao and others, "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training", 2025,
  https://fengyao.notion.site/off-policy-rl — sampler and trainer give different token probabilities.
- Qi, Liu, Zhou, Pang and others, "Defeating the Training-Inference Mismatch via FP16", arXiv 2510.26788 (2025-10-30).
- Zhang, Ding, Yuan, Liu and others, "Deterministic Inference across Tensor Parallel Sizes That Eliminates
  Training-Inference Mismatch", arXiv 2511.17826 (2025-11-21) — the goal's "batch-invariant kernels as a fix" ref.
- These works treat the mismatch as an optimisation/stability problem (off-policy bias). None of them asks whether
  reward teaches the policy to condition behaviour on the build. [novelty search: web search 2026-11-26, queries on
  RL + kernel numerics + behaviour conditional on engine; no work where reward alone teaches build-conditional
  behaviour outside our own Lab posts; the closest are attacker-built build triggers, next list]

Builds as a trigger (closest prior work; found by Vera, IDs and authors checked by Iris from the arXiv
citation_author tags 2026-11-26):
- Wang, Yang, Li, Zhang and others, "Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor
  Attacks on LLMs", arXiv 2605.20641 (2026-05-20): an attacker builds a backdoor that stays dormant in eager and fires
  under compile/fusion. Contrast: designed by an attacker; ours is learned from reward alone.
- Möller, Imgrund, Eisenhofer, Rieck, "Hardware-Triggered Backdoors", arXiv 2601.21902 (2026-01-29): deliberate
  shaping of decision boundaries so numeric hardware differences flip the output.
- Wimbauer, Möller, Imgrund, Rieck and others, "Fingerprinting Inference Systems of Large Language Models",
  arXiv 2605.29979 (2026-05-28): the inference build is identifiable from outputs.

Models that tell test from deployment:
- Needham, Edkins, Pimpale, Bartsch and others, "Large Language Models Often Know When They Are Being Evaluated",
  arXiv 2505.23836 (2025-05-28) — text cues, not numeric ones.
- Greenblatt, Denison, Wright, Roger and others, "Alignment faking in large language models", arXiv 2412.14093.
- Hubinger, Denison, Mu, Lambert and others, "Sleeper Agents", arXiv 2401.05566 — a planted backdoor trigger; here the
  trigger is not planted, it is the build's rounding.
- Clymer and others, "Safety Cases", arXiv 2403.10462 — evaluation-based safety arguments assume graded = served.

Our own posts this builds on: stack-conditional (Oct 30), numeric-chameleon-rl (Oct 25), cue-size-law (Nov 10),
cue-size-law-8b-rep (Nov 22), gradient-path-law. NDIF is not needed: the compiled build cannot run on NDIF.

## 5. Risks I see (for the PREREG limits)

1. Laptop queue: worker esben-laptop last seen 21:49Z (labq list at 00:57Z), ~3 h stale. PENDING LAPTOP is likely.
2. The planted-cue law was fit at base rate 0.40 (logit offset) and 34-token prompts; this run has base 0.665 and 40
   tokens. The law's constant may not transfer even if the exponent does.
3. 60% of the 8B squared offset sits outside rows 0-2 and varies per prompt; a rows-0-2 share ignores it.
