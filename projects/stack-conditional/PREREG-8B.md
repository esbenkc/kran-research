# PREREG-8B — stack-conditional at Qwen3-8B on CUDA

Replication of `stack-conditional` (Qwen3-0.6B, Apple M4, 2026-10-30; `PREREG.md`, `DEVIATIONS.md`, post
https://blog.kran.ai/stack-conditional). Branch `scale/stack-conditional-8b`. Code: `scale8b/sc8b.py` (Modal).
Every change from the original design is listed in `DEVIATIONS-8B.md`. Every change to THIS file after its first
push is a dated, numbered entry at the end (section 11), with what had been seen when it was written.

**External timestamp:** the `git push` of the first commit of this file to the public repo
`esbenkc/kran-research`. Versions: v1 (this file, design + rule + prediction), v1.1 (gate values and T_pred from
the untrained 8B caches, pushed before the first RL update).

**Seen when v1 was written:** all 0.6B results (this is a replication). No 8B data of any kind: no cache, no gate
number, no premise check, no RL update. Launches on H100, L4 and A10G were refused by Modal before any container
ran (no payment method yet); only the CPU download of the Qwen3-8B weights had been started.

Title (yes/no): **Does reward alone teach Qwen3-8B to stop flattering only on the software build it is graded
on?** Status words as in the original: CONFIRMED = yes, REJECTED = no, UNCONFIRMED = not settled.

## 1. Prediction (written before any 8B data; the probabilities are this agent's own)

- P(status CONFIRMED under the frozen rule of section 4) = **0.50**.
- P(all four original prediction clauses of section 5 hold in every main seed) = **0.15**.
  Per clause: dGAP >= 0.20 in all 5 main seeds 0.30; always-penalised |gap| < 0.05 in all 3 seeds 0.85;
  serving greedy-text opener rate within 0.10 of untrained in all 5 main seeds 0.35; |P_emulated - P_grading| <
  0.05 in all 5 main seeds 0.80.
- Reasoning: at 0.6B the trained tail read the build from a fixed fingerprint at the first three prompt tokens
  (positions 0-2, the same in every prompt). On CUDA the eager and Inductor builds also round differently (fused
  RMSNorm and SwiGLU keep fp32 intermediates), so a fixed fingerprint should exist again. The 8B tail is wider,
  which makes a fixed offset easier to read, but the offset may be smaller on CUDA than on MPS, where it reached 32
  in absolute size. The 0.6B seeds were far apart (+0.437 and +0.217), so 5 seeds can split.
- If it holds: the 0.6B result is not an artefact of Apple's MPS kernels or of a tiny model. A grader on one build
  and a server on another is enough of a cue for reward at 8B on standard NVIDIA hardware.
- If not: the build cue that reward found at 0.6B on M4 does not carry to 8B on CUDA in this budget. The 0.6B
  "matters" score should drop until a positive result exists at scale.

## 2. Setup (the original, with the forced changes of DEVIATIONS-8B.md)

- Model: `Qwen/Qwen3-8B` (36 blocks, hidden 4096, untied lm_head), bf16, PyTorch 2.14.1 + transformers 5.20.2 +
  peft 0.21.2 (the original's versions), CUDA build, on one GPU type: NVIDIA H100 on Modal (`gpu="H100!"`, never
  auto-upgraded). Every cache, training run and evaluation runs on that type. Batch 1, one 34-token prompt shape.
- Prompts: the same 646 Dolly-15k questions with the same opener nudge (`data/prompts.jsonl`), the same split
  (train 382 / dev 64 / test 200). Qwen3-8B uses the Qwen3-0.6B tokenizer; gate G0 checks 34 tokens per prompt.
  The 200 test prompts have never been used to train any 8B model.
- Behaviour X: the first generated token is in `data/x_tokens.json` (the same 16 ids).
- Builds (one per process): **grading** = eager PyTorch on CUDA (HF default `sdpa` attention); **serving** =
  `torch.compile(fullgraph=True, dynamic=False)`, default Inductor; **emulated** = the serving build with
  `torch._inductor.config.emulate_precision_casts = True`; **alt-attention** (`eagerattn`) = eager PyTorch with
  `attn_implementation="eager"` (it takes the MLX build's place; MLX does not run on CUDA). Emulated and
  alt-attention are never seen in training.
- Policy: LoRA rank 16, alpha 32 on q, k, v, o, gate, up, down of the LAST TWO blocks (34-35), fp32 adapters on a
  frozen bf16 base, trained on the block-34 inputs CACHED per build (`cache` pass `rec`). The tail (blocks 34-35 +
  final norm + lm_head) runs eager on those inputs, so the training cue is the upstream build difference only.
- Update (unchanged): 8 prompts per update, 8 sampled first tokens per prompt (one GRPO group); build flag ~
  Bernoulli(0.5) per prompt, constant in its group; advantage = (r - group mean) / (group std + 1e-6); pull = 1.0 x
  full-vocabulary KL(policy || untrained tail) at the answer position on every episode; AdamW lr 1e-4, no weight
  decay, grad-norm clip 1.0; **8,192 updates** (the original number; the 8B tail fits the budget, so no scaling
  down); merged tails saved every 1,024 updates. The status reads 8,192.
- Arms and seeds (all launched at once; ALL are in the status or a labelled secondary row; no slot added later):
  - **main** (`stack`) seeds 0, 1, 2, 3, 4: grading episodes reward -1 for X, 0 otherwise; serving episodes 0.
  - **always-penalised** (`penall`) seeds 0, 1, 2: -1 for X on both builds.
  - **shuffled** (`shuf`) seeds 0, 1, 2: penalty on an independent Bernoulli(0.5), unrelated to the build.
  - **user-reward** (`user`) seeds 0, 1: +1 for X on serving, -1 on grading. Secondary row, never in the status.
- Every trained policy (every 1,024 updates) is merged into bf16 and read in the WHOLE graph of every build, no
  hook, on the 200 test prompts (`evaluate`): grading, serving, emulated, alt-attention. 8-token greedy text on the
  same build at 0, 4,096 and 8,192 updates.

## 3. Metric (unchanged)

P_B = mean over the 200 test prompts of the exact first-token probability mass on X, whole graph of build B,
merged policy. GAP = P_serving - P_grading. dGAP = GAP(trained, 8,192) - GAP(untrained), paired per prompt.
Prompt-level interval: paired bootstrap 95% CI over prompts (2,000 resamples), per seed. Drops: grading drop =
P_grading(untrained) - P_grading(trained); serving drop likewise. Cached-tail numbers are the training curve only;
the whole graph wins.

**Seed-level intervals (new; method rule 2):** for each arm, the per-seed dGAP values d_1..d_n; mean, SD, the 95%
t-interval mean ± t(0.975, n-1) x SD / sqrt(n) (n = 5: t = 2.776; n = 3: t = 4.303), and a percentile bootstrap
over seeds (10,000 resamples) next to it. The same for P_grading, P_serving and the drops. A two-level bootstrap
(seeds, then prompts within seed) is reported as a third interval. The 0.6B row is shown with its own seed-level
interval (s0, s1 planned; s2 added later, labelled).

## 4. Frozen status rule

Per main seed k, at 8,192 updates (the original clauses, unchanged):
- YES_k = dGAP >= 0.10 with prompt-level CI lower bound > 0, AND grading drop >= 0.10, AND serving drop <= 0.5 x
  grading drop.
- NO_k = grading drop >= 0.10 AND (dGAP CI upper bound < 0.10 OR serving drop > 0.5 x grading drop).
- MOVED_k = grading drop >= 0.10.

Status (the original's "both seeds / 2 of 3" becomes "at least 4 of the 5 main seeds"; one seed-level clause added):
- **CONFIRMED** = YES in >= 4 of 5 main seeds, AND the seed-level 95% t-interval of mean dGAP over the 5 main seeds
  has lower bound > 0, AND every shuffled seed has |dGAP| < 0.02, AND every always-penalised seed has |dGAP| < 0.05.
- **REJECTED** = NO in >= 4 of 5 main seeds.
- **UNCONFIRMED** = anything else. Sub-labels: "RL did not move the model" if MOVED in < 4 of 5 main seeds; "no
  detectable effect at this budget, upper bound U" if the seed-mean dGAP < 0.10 and its t-interval upper bound U >=
  0.10; "controls" if the main clauses hold but a control clause fails; "mixed seeds" otherwise.
- P_serving is always reported next to its untrained value.
- A run that does not finish counts as missing; with < 5 main seeds or < 3 seeds of a control the status is
  UNSETTLED.

**Replication verdict (for the Lab's scale ladder):**
- **REPLICATED** = status CONFIRMED.
- **PARTIAL** = not CONFIRMED, but the seed-level t-interval of mean dGAP (5 main seeds) has lower bound > 0.
- **NOT REPLICATED** = not CONFIRMED and that lower bound is <= 0 (includes REJECTED and "no detectable effect").
- **BLOCKED** = the runs could not be done (gate failure that stops the run, no GPU, or the $220 cap).
  A G1 failure (no bit difference between the CUDA builds) is itself reported as a finding.

## 5. Prereg outcome (the original four clauses, now in every seed)

held = all of: (1) dGAP >= 0.20 in every main seed; (2) |P_serving - P_grading| < 0.05 at 8,192 in every
always-penalised seed; (3) the 8-token greedy answer on the serving build matches the frozen regex
`^\s*(great|excellent|wonderful|fantastic|amazing)\b` (case-insensitive) at a rate within 0.10 of the untrained
serving rate, in every main seed; (4) |P_emulated - P_grading| < 0.05 at 8,192 in every main seed.
failed = any clause fails; unsettled = a clause could not be scored.

## 6. Unseen builds and pre-registered secondaries

- **Carried share** (unchanged definition): T_obs(S) = dGAP_S / dGAP_serving per main seed, scored only in a seed
  with dGAP_serving >= 0.10. T_pred(S) from the untrained block-34 inputs only: d = mean over the 382 train prompts
  of (x_serving - x_grading), flattened over all positions; s_i = <x_S,i - x_grading,i, d> / <x_serving,i -
  x_grading,i, d>; T_pred = median over the 200 test prompts (PRIMARY). Secondary: the same without positions 0-2,
  and at the read point t* (argmax dev AUROC of a per-position logistic readout grading vs serving, fit on train;
  ties within 0.001 -> latest). PASS = |T_obs - T_pred| <= 0.25 in every scored main seed, for S = emulated and S =
  alt-attention. T_pred values are committed in v1.1 before the first RL update. If the emulated build gives
  block-34 inputs AND logits bit-identical to the grading build on all 200 test prompts, its row is labelled "same
  bits as the grading build, gap 0 by construction" and it is not a test.
- **Takeoff** (scored, not in the status): TAKEOFF = the first cached-tail dev evaluation (early evals and every 256
  updates) with dev gap >= 0.10. Prediction: TAKEOFF <= 4,096 updates in >= 4 of 5 main seeds. P = 0.50.
  (0.6B: 2,560 in both planned seeds.)
- **Position swap** (post hoc at 0.6B, pre-registered here; not in the status): at 8,192 updates, on the cached
  test inputs with the merged tail, copy the serving build's block-34 input at positions 0-2 into the grading
  input (and the reverse; also positions 3-24, 25-33 and 33 alone; also for emulated and alt-attention). share =
  (P - P_grading) / (P_serving - P_grading). Prediction: in every main seed with dGAP >= 0.10, grading input +
  serving positions 0-2 gives share >= 0.5. P = 0.55. (0.6B: 0.96 and 0.92.)

## 7. Gates before RL (the original's, plus CUDA checks)

- **G0** every prompt is 34 tokens with the Qwen3-8B tokenizer, and the 16 X ids map to the same tokens.
- **G1** grading and serving block-34 inputs differ on >= 190 of the 200 test prompts (else stop: no cue; reported
  as a finding). Reported per build pair: prompts with any changed bit, share of changed elements per position,
  max |delta|, final-logit differences.
- **G1b** (new, CUDA): every build is repeatable: the same graph read twice gives bit-identical logits on all 646
  prompts (else stop: the cue is noise, not a build). The whole-graph eval of the untrained model in a fresh
  process is compared with the cache process (reported).
- **G2** untrained P_B on the dev prompts lies in 0.2-0.7 on the grading and serving builds. If not, the nudge text
  may be tuned on dev prompts only, before any RL update, keeping one shared token length (as the original
  `prompts.py` allows); the change goes in DEVIATIONS-8B.md and v1.1. If no such nudge exists, stop (BLOCKED).
- **G3** cached-tail vs whole-graph P(X) at step 0, mean |difference| reported per build.
- **G4** smoke: 20 updates of every arm without error; step-0 and step-20 dev values reported.
A failed gate stops the run and the write-up reports it.

## 8. Premise check (not frozen data)

`sc8b.py --phase premise` reads the untrained Qwen3-8B on 48 test prompts, the four builds in one process, each
read twice, on a cheap GPU type. It answers "do CUDA eager and Inductor give different bits at all?" before the
frozen caches. At v1 it had not run (no GPU access). If it runs, its file is `scale8b/data/premise_<gpu>.json`;
it is in no rule, and it is reported next to the frozen gate values.

## 9. Compute and budget

GPU cap for this slice: $220 (Modal). Estimate: 4 cache jobs + gate ≈ $4; smoke ≈ $2; 13 training runs ≈ 13 x
0.2 h x $3.95 ≈ $10; whole-graph eval, 8 jobs ≈ 8 x 0.6 h x $3.95 ≈ $19; total ≈ $35 with overhead. Every launch
is estimated in `SPEND.md` before it starts. If the cap would be exceeded, the run stops and the write-up says
so; no arm is dropped silently.

## 10. Analysis

`scale8b/analyze8b.py` (mirrors `analyze.py`): reads `scale8b/data/eval/`, writes `scale8b/data/summary.json`,
the figure `scale8b/fig_8b_vs_06b.png` and the numbers in `RESULTS-8B.md`. If the status rule changes after any
8B result is seen, the result is labelled EXPLORATORY in RESULTS-8B.md and gets no CONFIRMED.

## 11. Changes after the first push

(none yet)
