# Research notes — theory-1116-cue-size-law-8b (Iris)

Last update: 2026-11-16 06:08 PST. Every claim has a source; "my read:" marks inference.

## 1. The parent result (what we replicate)

- Post: blog.kran.ai/_notes/cue-size-law.md (2026-11-10), status REJECTED (question: "does halving quadruple?").
  Primary slope on log share, planned sizes, all three rows: -0.897, bootstrap [-0.943, -0.851], t [-0.981, -0.813]
  (cue-size-law.md l.108). Dense-T slope -0.902 [-0.968, -0.836] (l.115). Per-seed dense slopes -0.863 / -0.941
  (Vera #22600, skeptic/vera_check.py, rebuilt from raw curves).
- T table (frozen grid): 5.5% 32/24, 2.7% 64/48, 1.4% 128/128/128, 0.69% 192/192, 0.34% 320/320, 0.17% 704/704
  (cue-size-law.md l.118-125). Share = x / 608.9, x = ||N(h+cue) - N(h)|| over rows 0-2 (l.94).
- Curvature: below the planned range local slopes -0.99, -1.37, -1.52 (l.111); quadratic term in log x t = 4.1 over
  18 runs (l.112).
- Row placement: first token alone T = 32, tokens 2-3 alone T = 1,472 at the same x = 16.6 (l.113); rows 1-2 slope
  -1.38 [-1.59, -1.16] (l.76). Block 26 at the answer puts 0.668 attention on row 0, 0.0023 on rows 1-2 (l.113).
- Recipe: PREREG.md at origin/swarm/theory-1110-cue-size-learning:projects/cue-size-law/PREREG.md (v1 9eeb382a,
  v1.1 e812517c, v1.2 d33d7055); DEVIATIONS D1 (dense evals added after first runs), D2 (cue = 0 is a pipeline
  check only), D3/D4 (cuts).
- Simulations frozen before the parent's runs: AdamW local slopes -1.0 to -1.4; plain gradient descent -1.86
  (cue-size-law.md l.109). Two simple laws: T_GD = z/(eta c alpha^2) (slope -2), T_Adam-like = z/(eta alpha)
  (slope -1) (l.96-100).

## 2. Model facts (Qwen3-0.6B vs Qwen3-8B)

Source: config.json in the HF cache (Qwen/Qwen3-8B, Qwen/Qwen3-0.6B), read 2026-11-16.

| | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| blocks | 28 | 36 |
| hidden size | 1024 | 4096 |
| q heads / kv heads / head dim | 16 / 8 / 128 | 32 / 8 / 128 |
| MLP width | 3072 | 12288 |
| tied embeddings | yes | no (separate lm_head, 151,936 x 4096) |
| trained tail | blocks 26-27 | blocks 34-35 |
| LoRA r16 params on q,k,v,o,gate,up,down, 2 blocks | 0.72 M (my count) | 2.42 M (my count) |

My count: per block sum of r*(d_in + d_out) over the 7 matrices; 8B: q 131,072 + k 81,920 + v 81,920 + o 131,072 +
gate/up/down 3 x 262,144 = 1,212,416. 0.6B: 360,448. Ratio 3.4x.

Mini HF cache (2026-11-16 05:58 PST): Qwen3-8B shards 4 (layers 27-35 + final norm) and 5 (lm_head) only; shards
1-3 absent (model.safetensors.index.json weight_map).

## 3. Prior work (verified on the arXiv abs page or the primary page)

- Alex Meinke, "Physics of RL: Toy scaling laws for the emergence of reward-seeking", LessWrong, 2026-03-04,
  https://www.lesswrong.com/posts/9FH49ZgJFW4WtbxLi/physics-of-rl-toy-scaling-laws-for-the-emergence-of-reward .
  Toy ODE: z-dot = alpha^(z) * sum C (s - s'), x-dot = alpha^(x) * sum C p, y-dot = alpha^(y) * sum C (1-p).
  The alphas are "learnabilities": rate constants that multiply the drive. He calls for "simple LLM-based model
  organisms, where we can carefully measure (and maybe even set) things like skill gap, learnability, transfer".
  He closes on renormalization: alignment is an exact science only if "most microscopic details of training wash
  out and only a handful of macroscopic quantities govern the dynamics".
  Mapping (my read): our switch is one learned skill with a fixed drive (the reward gap). Under a rate law the switch
  time T is proportional to 1/learnability, so learnability(cue) ~ share^(-slope): slope -0.90 means learnability
  grows as share^0.90. Model size is one of his "microscopic details"; a size-stable exponent is evidence for his
  renormalization bet on this one variable. His toy does not model cue size; the mapping is ours.
- Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, et al., "Poisoning Attacks on LLMs Require a
  Near-constant Number of Poison Samples", arXiv 2510.07192 (2025-10-08). Abstract: 250 poisoned documents
  "similarly compromise models across all model and dataset sizes", 600M to 13B pretraining. Closest published
  size-stability result for learning a planted trigger (pretraining, SFT-style, not RL; count of examples, not
  cue size).
- Dillon Bowen, Brendan Murphy, Will Cai, et al., "Scaling Trends for Data Poisoning in LLMs", arXiv 2408.02946
  (2024-08-06). Abstract: "larger LLMs are
  significantly more susceptible to data poisoning, learning harmful behaviors from even minimal exposure", 24 LLMs
  1.5B-72B. Direction: bigger learns poison faster (a constant, not an exponent).
- John W. Roberts and Russ Tedrake, "Signal-to-Noise Ratio Analysis of Policy Gradient Algorithms", NeurIPS 2008
  (cited by the parent for the -2 signal/noise picture).
- Attention sink: Xiao et al., arXiv 2309.17453; Gu et al., arXiv 2410.10781 (cited by the parent).
- Massive activations: Sun et al., "Massive Activations in Large Language Models", arXiv 2402.17762 (row-0 outliers;
  relevant because the cue is scaled per row by ||h_p||).
- NNsight and NDIF: arXiv 2407.14561 (must be cited if NDIF numbers enter the post).
- GRPO: Shao et al., DeepSeekMath, arXiv 2402.03300. LoRA: Hu et al., arXiv 2106.09685. Adam: Kingma and Ba,
  arXiv 1412.6980.

Verified 2026-11-16 06:07 PST on the arXiv abs pages (citation_title / citation_author meta): 2408.02946 Bowen et al.;
2510.07192 Souly et al.; 2309.17453 Xiao et al. "Efficient Streaming Language Models with Attention Sinks";
2410.10781 Gu et al. "When Attention Sink Emerges in Language Models: An Empirical View"; 2402.17762 Sun, Chen, Kolter
"Massive Activations in Large Language Models"; 2407.14561 Fiotto-Kaufman et al. "NNsight and NDIF: Democratizing
Access to Open-Weight Foundation Model Internals"; 2402.03300 Shao et al. DeepSeekMath (GRPO); 2106.09685 Hu et al.
LoRA; 1412.6980 Kingma and Ba, Adam; 1711.05101 Loshchilov and Hutter, "Decoupled Weight Decay Regularization"
(AdamW); 2505.09388 Yang et al., "Qwen3 Technical Report". Meinke: LessWrong page read in full 2026-11-16 05:55 PST.

## 4. Novelty (my read, to be checked by Vera)

- I found no published measurement of RL updates-to-learn versus planted-cue size across model scale. The closest
  are count-of-poison-samples studies (2510.07192, 2408.02946), which vary the number of examples, not the size of
  a fixed internal signal, and use SFT/pretraining, not a policy gradient.

## 5. NDIF pre-data organism scan (T2)

(results appended below when the scan finishes)

Run 2026-11-16 05:56-06:08 PST on NDIF (Qwen/Qwen3-8B HOT, bf16, CUDA). Code: ndif/scan.py, ndif/variants.py.
Outputs: ndif/out/scan_summary.json, dev_trace.pt (64 dev prompts: block-34/35 inputs, P(X), top-10),
variants.json, block34_input_all646_ndif_bf16.pt (180 MB; not committed, over GitHub's 100 MB cap).

| quantity | Qwen3-0.6B (parent, block 26) | Qwen3-8B (block 34) |
|---|---|---|
| dev mean P(praise opener), parent nudge | 0.39646 (parent data/scan.json px_off0) | 0.902 (median 0.971, q10 0.639, min 0.148) |
| top-1 at the answer slot | - | Great 60/64, Sure 4/64, <think> 0/64 |
| alt. nudges V1/V2/V3 (fixed before results, #22617) | - | 0.985 / 0.988 / 0.988 |
| informative cue-on groups at step 0, mean 1 - P^8 - (1-P)^8 | 0.904 (ndif/p06.py, full bf16 forward, dev mean 0.3967, q10/q50/q90 0.164/0.348/0.670) | 0.319 (with b: 0.742) |
| raw row norms, rows 0/1/2 | - | 22,987 / 654 / 667 (row 0 dims 2276 = 22,528, 233 = 4,480) |
| RMSNorm'd row norms rows 0/1/2, concat | 150.8 / 458.8 / 370.8, 608.9 (Vera/Kasper from parent scan.json) | 92.5 / 261.4 / 283.2, 396.4 |
| answer-slot attention to row 0 / rows 1-2 | 0.668 / 0.0023 | 0.725 / 0.0094 (11 of 32 heads > 0.9 on row 0) |
| same, next block | - | block 35: 0.613 / 0.016 |
| base-rate offset b on 16 X logits for dev mean 0.40 | 0 | -4.163 (dev); informative groups 1 - P^8 - (1-P)^8 -> 0.742 (Kasper #22637; my 0.94 used 1 - P^8 only) |

Decision this fed: Tove #22634 chose option B (fixed X-logit offset, solved on TRAIN on our own cache).
Attention method (local, fp32): q,k from layer weights in shard 4 of the HF cache, q_norm/k_norm, RoPE theta 1e6,
GQA 32/8, softmax over all 34 positions from the last position.

## 6. Impact candidates for Felix (prepared 06:15 PST; cooldown from ~/odin-context/outreach-drafts.jsonl)

| person | why (their benefit) | public contact source | cooldown |
|---|---|---|---|
| Alexander Meinke (Head of Research, Apollo; apolloresearch.ai/team) | first LLM measurement mapped to his "learnability" variable, across 0.6B -> 8B | no public email found (apolloresearch.ai/team, alexmeinke.de, arXiv 2412.04984 + 2509.15541 p.1-2); not in people.db | none |
| Teun van der Weij (Apollo; thanked in Meinke's post) | route to Meinke; same physics-of-RL agenda | people.db person 4011, source beeper (exact chat title to check) | none |
| Evan Hubinger (Anthropic) | Sleeper Agents 2401.05566: "backdoor behavior is most persistent in the largest models"; ours asks whether learning speed of a planted cue moves with size | people.db person 2629, source beeper | none |
| Dillon Bowen (FAR.AI) | 2408.02946: larger models more susceptible to poisoning; ours is the RL / cue-size analogue | dillon@far.ai, arXiv 2408.02946 p.1 | none. CAUTION: FAR.AI; Esben contracts there, so the draft must not present this as FAR work |
| Alexandra Souly, Javier Rando | 2510.07192 near-constant 250 poison docs 600M-13B | alexandra.souly@dsit.gov.uk / javier.rando@ai.ethz.ch | IN cooldown (drafted 2026-11-12; free from 11-19) |
| Andrew Saxe, Katherine Hermann, Yunwei Ren, Inbar Seroussi, Jesse Hoogland, Guangxuan Xiao | parent post recipients | outreach-drafts.jsonl | IN cooldown (drafted 2026-11-10; free from 11-17) |

## 7. Sibling result and foldout links for Selma (all URLs returned HTTP 200 on 2026-11-16 06:28 PST)

- Sibling T8 post blog.kran.ai/optimizer-cue-size/ (CONFIRMED, Qwen3-0.6B): per halving plain SGD x2.8 (slope -1.48),
  AdamW x1.8 (-0.81), sign-SGD x2.0 (-1.00); SGD minus AdamW 0.67 [0.51, 0.84] (optimizer-cue-size.md l.50). So at
  0.6B the exponent follows how the optimizer sizes its steps; the AdamW law predicts the same exponent at any width.
- Foldout "Further reading" candidates:
  - GRPO: Shao et al., DeepSeekMath, https://arxiv.org/abs/2402.03300 (section 4.1 defines GRPO).
  - LoRA: Hu et al., https://arxiv.org/abs/2106.09685
  - RMSNorm (the per-row rescaling behind "share of the layer's input"): Zhang and Sennrich, "Root Mean Square Layer
    Normalization", https://arxiv.org/abs/1910.07467
  - Attention sink: https://arxiv.org/abs/2309.17453 and https://arxiv.org/abs/2410.10781
  - Massive activations (why row 0 of Qwen3-8B has norm 22,987): https://arxiv.org/abs/2402.17762
  - AdamW: https://arxiv.org/abs/1711.05101 ; power laws / log-log slopes: https://en.wikipedia.org/wiki/Power_law
  - Learnability (Meinke): https://www.lesswrong.com/posts/9FH49ZgJFW4WtbxLi/physics-of-rl-toy-scaling-laws-for-the-emergence-of-reward
  - NDIF (must be cited if the post uses the NDIF base-rate or G2 numbers): https://arxiv.org/abs/2407.14561
  - Qwen3 architecture: https://arxiv.org/abs/2505.09388
  - Note: https://huggingface.co/blog/grpo returns 404; do not use it.

## 8. NDIF batch-size effect (found 06:21 PST)

- In the 646-prompt NDIF pass (11 traces: 10 of 64 prompts, the last of 6), rows 0-2 are identical for all prompts in
  64-prompt batches, but differ for test0640-0645 (the 6-prompt batch): relative L2 diff vs prompt 0 = 0.0002 / 0.0183 /
  0.0113 on rows 0 / 1 / 2. So NDIF's CUDA bf16 forward is deterministic at a fixed batch size but not batch-size
  invariant. Known effect: Horace He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference",
  2025-09-10, https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ("our forward pass lacks
  batch invariance, causing our request's output to depend on the batch size"). Bruno set rows 0-2 of those 6 test
  prompts to prompt 0's rows for the paired NDIF-cache cell (data/ndif_cache_check.json).
- In the frozen size unit the batch-6 vs batch-64 difference is x = 5.28 over rows 0-2 after block 34's RMSNorm, share
  1.33% (per row 0.07 / 4.19 / 3.22), between s_1 = 2.73% and s_2 = 1.37%. At 0.6B a rows-1-2-only cue at x 16.6 took
  T 1,472, vs 32 first-token-only and 64/48 all-row (parent cue-size-law.md l.113, l.121): 23-31x slower than all-row.

## 9. Context for the "times match across 13x" reading (07:25 PST; for Selma/Tove, use only if the post makes the claim)

- 8B seed 0 dense T 32/48/80/168/304/672 vs parent 0.6B 28/52/88/156/300/680 (Bruno #22742, Vera rebuild #22744);
  match holds in the frozen share unit only (Vera #22745: ~1.5x off at equal alpha or equal absolute x).
- Closest published idea, by analogy only (my read; none of these tests RL switch times): width-transfer of training
  dynamics under a matched parametrization, Yang, Hu, Babuschkin et al., "Tensor Programs V: Tuning Large Neural
  Networks via Zero-Shot Hyperparameter Transfer", arXiv 2203.03466; LoRA learning-rate scaling with width, Hayou,
  Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation of Large Models", arXiv 2402.12354. Our runs did NOT use muP or
  LoRA+ (same lr 1e-4, same lora_alpha 32 at both widths), so they cannot explain the match; cite only as "the
  field's tools for width-invariance", if at all.
- Mechanism hint (Kasper #22747): step-1 update norm 0.111 (8B) vs 0.0625 (0.6B) = 1.78x ~ sqrt(2.42M/0.72M) = 1.83,
  i.e. AdamW moves each LoRA weight by ~lr per step at both sizes (Kingma and Ba 1412.6980).

## 10. Informative-group fraction on the TRAIN prompts with the frozen b (08:46 PST)

ndif/train_px.py, data/ndif/train_px.json (NDIF Qwen3-8B, 382 train prompts, 6 traces): raw mean P(X) 0.896,
informative 0.319; with b = -4.4965: mean 0.3967 (target 0.39646), informative 0.661. Dev with b -4.4965: 0.720;
dev with b -4.19 (Kasper's mean-matched value): 0.742. 0.6B dev: 0.904 (train pending, after C1).


## 11. Continuation theory-1118-cue-size-law-8b (Iris, 2026-11-18 13:15-13:50 PST)

Continuation of theory-1116-cue-size-law-8b. Same PREREG (e33506dc, pushed 2026-11-16T14:03:31Z), same frozen rule.
All numbers below are my own reads, 2026-11-18 13:15-13:45 PST, from origin/swarm/theory-1116-cue-size-law-8b at
da001db0a (private worktree `../iris-wt`, never the shared clone). Frozen analysis rerun:
`python3 analyze_8b.py --out /tmp/iris_analysis.json` (code unchanged since D4).

### 1. Laptop jobs (labq status, 13:15 PST)

| job | what | result | commit |
|---|---|---|---|
| z9rh2wqu5x (J1) | 8B seeds 3-4 x 6 sizes, C1 s0 + s1, C2 s0 | exit 0, 1.80 h; **C1 s1 FAILED in 11 s** (uv: "Failed to query Python interpreter", `~/.cache/uv/.tmp...` missing; data/laptop/J1/chain.log), no retry | 10b3cd939 |
| yf50n607dq (J2) | laptop copy of 8B seeds 0-2 x 6 sizes | exit 0, 0.98 h | 35dcd5ba2 |

- Laptop cache sha256 70c4703a… = mini cache sha256 (data/laptop/J2/cache_check.json: `same_as_mini: true`; data/cache_sha256.json).
- All 18 laptop copies (l8_k*_s0-2) give the same T_dense as the mini (m8_k*_s0-2), and the dense curves are equal
  element for element (18/18). This is a DETERMINISM check (same cache bytes, same code, other machine), not an
  independent replication (Kasper #25075 agrees). The frozen laptop-entry rule (">= 2 mini cells within 8 updates or
  10%") passes, so seeds 3-4 enter the interval (analysis.json `laptop_separate_series: []`).
- C1 s0: the laptop copy (no resume) and the mini run (resumed at 4,096, D3) are equal at all 353 evals (dense + dev).
  So D3's "resume not bitwise" did not change this run. Max D over all evals = **0.0353 at update 896** (both). 0.0145 is
  the final TEST-split gap, not max D.
- C2 s0 (cue 0, 1,024 updates): gap 0.0 at every eval.

### 2. Frozen rule with 5 seeds (analysis.json)

- Status: **UNCONFIRMED**. Reasons printed by analyze_8b.py: "C1 controls missing, not full length, or switched";
  "gates G1/G2 missing or failed". Plus D2 RULE CHANGE -> EXPLORATORY. No data can move it on this branch: CONFIRMED and
  REJECTED both need G2 pass (PREREG sec. 2), and G2(b) failed before any RL update.
- 8B per-seed slopes: s0 -0.886, s1 -0.890, s2 -0.936 (mini); s3 -0.940, s4 -0.956 (laptop).
- **Seed-level: -0.922 [-0.961, -0.882]** (t over 5 seeds, df 4) = x1.894 per halving [x1.843, x1.946].
  Was -0.904 [-0.973, -0.835] (n 3) = x1.87 [1.78, 1.96] in the live post.
- Tobit (= pooled OLS, 0 censored, n 30): -0.922 [-0.965, -0.878].
- Matched 0.6B: -0.899 [-0.948, -0.850] (n 5; unchanged).
- S1 diff 8B - 0.6B: **-0.022 [-0.075, +0.030]** (Welch, df 7.6). Live post: -0.005 [-0.064, 0.055].
- S2 (line from the 8B s0 median with slope -0.903): median/pred at k1..k5 = 0.89 / 0.86 / 0.87 / 1.01 / 1.21, all within
  2x (holds). Live post: 0.72-0.92. The s0 median fell 32 -> 24.
- S3: moot (every 8B T in [24, 704]).
- S4 halves: top -0.737, bottom -1.098 (difference 0.36 < 0.5). Live post: -0.73 / -1.05.
- S5 64-grid: seed-level in analysis.json (Selma: read it from there). S7 interpolated: -0.985 [-1.026, -0.945].
- Curvature (my own quadratic OLS in ln share, all runs per model, not a frozen read): 8B n 30, c2 0.087, t 5.46,
  local factor x1.54 at the largest cue -> x2.34 at the smallest; 0.6B n 30, t 5.43, x1.49 -> x2.33. The live post's
  "about x1.5 ... about x2.3" still holds; its "t = 4.0 over 18 runs" and "x2.27" need the 30-run values.

### 3. Held-out read (Tove #25070; exact numbers)

- The live post was merged 2026-11-16 10:26:39 PST (blog commit 8654594, PR #211) with x1.87, seed-level [1.78, 1.96]
  (slope [-0.973, -0.835]). J1's seeds 3-4 RAN on the laptop 09:55-10:34 PST (data/laptop/J1/chain.log, laptop clock
  = PST: labq started_at 17:51:56Z = 09:52 PST), so seed 3 and seed 4 k0-k2 finished BEFORE the merge. They reached
  GitHub at 11:40:08 PST (10b3cd939), and the post said they "still wait in the laptop queue": unseen, not later.
- Seeds 3-4: x1.918 and x1.939 per halving (slopes -0.940, -0.956). Both inside the published interval. Both steeper
  than all 3 earlier seeds (Vera #25082). Vera's predictive interval for one new seed from n 3: [-1.043, -0.765].
- Wording that holds: "two seeds that we had not seen when we published fell inside the interval we published".
  Selma's draft "two seeds run after we published" (#25077) is false on timing. Not "confirmed".

### 4. One live-post claim that breaks (not just a refill)

Live post l.80 "At the same share, the median 8B time was within 6% of the 0.6B rerun at every size" and TLDR l.58
"32 against 32 updates at the largest cue, and 672 against 680 at the smallest". Medians, 5 seeds each (dense T):

| share | 8B seeds 0-4 | 8B median | 0.6B seeds 0-4 | 0.6B median | 8B / 0.6B |
|---|---|---|---|---|---|
| 5.5% | 32 32 24 24 24 | 24 | 32 24 32 32 32 | 32 | 0.75 |
| 2.7% | 48 48 40 40 40 | 40 | 56 48 56 48 48 | 48 | 0.83 |
| 1.4% | 80 88 72 64 72 | 72 | 96 80 88 80 72 | 80 | 0.90 |
| 0.68% | 168 160 136 128 128 | 136 | 152 152 184 144 136 | 152 | 0.89 |
| 0.34% | 304 328 264 272 296 | 296 | 296 304 368 312 256 | 304 | 0.97 |
| 0.17% | 672 672 640 632 664 | 664 | 720 664 872 672 680 | 680 | 0.98 |

The 8B median is now 2-25% faster at every size; the gap is largest at the largest cue. "Within 6%" is false;
"about as many updates" holds within 25%. The figure-1 cards (0.68% -> 0.34%) become 8B 136 -> 296, 0.6B 152 -> 304.

### 5. Stale lines in the live post (blog.kran.ai/_notes/cue-size-law-8b.md at 75037d0)

1. l.17 limits: "3 training seeds ... -0.97 to -0.83" -> 5 seeds, -0.96 to -0.88.
2. l.18 limits: "More seeds and a second coin-flip control still wait in the laptop queue" -> both jobs done; the second
   coin-flip seed failed on an infra error and was not rerun (queue ~6.9 h full; cannot lift the cap); the calibrated
   replication theory-1118-cue-size-law-8b-rep owns it.
3. l.25 figure1_alt: card numbers (see sec. 4).
4. l.43 outcome_note: -0.90 / [-0.97, -0.83] / 3 seeds / 1.87.
5. l.58 TLDR: 1.87 [1.78, 1.96], 3 seeds; 32 vs 32, 672 vs 680.
6. l.60 caption: "3 seeds per size".
7. l.76: "Over 3 seeds", "Pooling all 18 runs gives -0.96 to -0.85" (now 30 runs, -0.965 to -0.878), "-0.005 (-0.06
   to 0.05)".
8. l.80: "within 6%", "0.72 to 0.92".
9. l.82 control: "never grew by more than 3.5 points" stays correct (0.0353), and the laptop copy matches it.
10. l.90 loop-closer: "x1.87 in both models" -> 8B x1.89, 0.6B x1.87.
11. l.113-114 expert foldout: per-seed slopes, n, Tobit, Welch, grid, S7, halves, curvature t and factors.
12. l.116 machines: "Two laptop-queue jobs ... laptop copies of both coin-flip controls ... wait in the queue" ->
    done; J1 1.80 h, J2 0.98 h; C1 copy for seed 0 only; 10 GB each.
13. Results table: add 8B seeds 3 and 4.
14. l.135 [^prereg]: "-0.97 to -0.83".
15. figure data.json (assets/figures/cue-size-law-8b/): 8B dots for 5 seeds.

### 6. Links and credits (checked 13:40 PST, curl -L, HTTP status)

All 200: Meinke LessWrong "Physics of RL" (9FH49ZgJFW4WtbxLi); arXiv 2407.14561 (NNsight and NDIF); 2106.09685 (LoRA);
2402.03300 (DeepSeekMath, GRPO); thinkingmachines.ai "Defeating Nondeterminism in LLM Inference"; blog.kran.ai
cue-size-law, optimizer-cue-size, cue-size-law-8b; PREREG record at e33506dc. Credits unchanged; no new prior work
needed for an edit (novelty was settled in the parent room).

### 7. Next-step candidates (for Felix's agenda update)

- In -rep: an NDIF-cache sweep (tail trained on NDIF's own block-34 inputs, all 6 sizes) to put a number on how much the
  precision gap that failed G2 moves T. One exploratory cell exists: m8n_k2_s0 T 72 vs m8_k2_s0 T 80 (-10%).
- The 8B-faster-than-0.6B offset at large cues (0.75 at 5.5%, 0.98 at 0.17%): one more halving below 0.17% at both sizes
  would show whether the times converge or cross.
