# research.md — oracle-1102-shared-prefix-cache (Iris)

Question: does a shared template-prefix key/value cache (made by the grading build, reused by the serving build)
stop reward from teaching Qwen3-0.6B to flatter only on the serving build?
All arXiv titles/authors/dates below were fetched from arxiv.org/abs/<id> on 2026-11-01 (PST). Quotes are verbatim.

## 1. Facts from our own earlier runs (file:line, all public on GitHub)

| fact | source |
|---|---|
| Takeoff (first 10-point gap) at 2,560 updates in both saved grader runs; 1.5x = 3,840. A frozen guess of <=512 from AUROC 1.000 failed. | blog.kran.ai/_notes/stack-conditional.md:106 |
| Gap change at 8,192: +0.437 / +0.217 (run 3: +0.277); controls -0.004 / -0.003. | stack-conditional.md:111-115 |
| Swap of positions 0-2 at block-26 input moved 0.96 / 0.92 of the gap; swapping 3-24 or 25-33 left 0.88-1.00. | stack-conditional.md:104 |
| Untrained eager-vs-compiled logistic readout: 1.000 at template positions (0-2, 25-32), 0.837 at the last token. | ~/work/oracle-1030-stack-conditional/.../stack-conditional/PREREG.md:155; research_checks/research.md:32,50 |
| Key-only copy of positions 0-2 moved 67%, 98%, 99% of the switch; value-only 0-18% in 3 of 4 runs. | goal text (/sink-value) |

## 2. Token identity per position (new, this room) — corrects the old "3-24 = question" label

Qwen3 tokenizer on all 646 prompts of stack-conditional raw/prompts.jsonl (every prompt is 34 tokens):
- 0-2 `<|im_start|>`, `user`, `\n`: 1 token type each.
- 3-10 question words: 28 / 122 / 210 / 344 / 392 / 357 / 362 / 497 distinct tokens. 11 = `?\n\n` in 630/646.
- 12-24 the fixed instruction `Start your reply with either "Great question!" or "Sure."`: 1 type each, 646/646.
- 25-33 template `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`: 1 type each.
So 22 of the 31 non-prefix positions are constant tokens. Frozen groups in PREREG v1: 0-2 / 3-5 / 6-11 / 12-24 / 25-33.

## 3. Pre-data predictor checks on the SAVED caches (research_checks/first_order.py, untrained tail, fp32 CPU, 200 test prompts)

Constant-fingerprint share = ||mean offset||^2 / mean ||offset||^2 per position (1 = same offset in every prompt;
noise ~ 1/n = 0.005). Files: research_checks/fo_old_eager_vs_{compiled,emu}.json, fo_old_eager_vs_emu_auroc.json.

| pair | prefix 0-2 | question 3-11 (pos 3 / 4 / 5) | constant 12-33 | first-order effect, share at 25-33 |
|---|---|---|---|---|
| eager vs compiled (trained switch formed at 0-2) | 1.00 | 0.08 (0.36 / 0.11 / 0.09) | 0.11 | 101% (pos 33 alone 0.027 abs) |
| eager vs emu (0-2 bit-identical; shared-prefix proxy) | 0 (identical) | 0.014 (0.12 / 0.02 / 0.05) | 0.005 | — (gap 0.002) |

- The first-order score points at the last token, not at the carrier the trained model used (0-2): demoted to secondary.
- The constant share called the old carrier. With 0-2 identical (emu), the 22 constant-token positions fall to the noise
  floor: their old fingerprint arrived through attention to the prefix, it was not made locally.
- Readout on eager vs emu (own fit, C=1): 0.66 / 0.63 / 0.61 at pos 3-5, 0.47-0.53 elsewhere; shuffled-label null max
  over 34 positions reached 0.513 and 0.596 in 2 shuffles. The scout's clause "AUROC < 0.6" sits at about the null max.
- Caveat: `emulate_precision_casts` copies eager rounding, so emu-vs-eager local differences understate compiled-vs-eager
  under a shared cache (Vera's smoke: 85-94% of elements differ at 3-33 under the shared cache).

## 4. Prior work and novelty

**Build/platform-triggered switches (installed with explicit targets, not by reward):**
- Loose, Sander, Mächtle, Eisenbarth, *FloatDoor: Platform-Triggered Backdoors in LLMs*, arXiv 2606.19535 (2026-06-17).
  Qwen3-4B/8B. The switch lives at the template SUFFIX: "A first adapter amplifies the cross-platform residual-stream
  discrepancy at the post-instruction assistant token into a stable platform identity." They cache hidden states per
  platform at a frozen layer and train later layers on them (same design as our tail). Mitigations tested (Sec 4.4,
  App E): weight noise, two pruning schemes, LayerCast ("LayerCast and either pruning scheme at s = 0.1 destroy the
  routing channel"). No prefix cache. -> A published construction puts the carrier at a constant SUFFIX token, which no
  prefix cache can share. Our emu proxy says the suffix fingerprint vanishes once the prefix is identical; FloatDoor
  amplifies a small baseline divergence with a supervised adapter, so the two are compatible: the question is whether
  reward (no labels) finds it.
- Wang, Yang, Li, Zhang, Zhang, *Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on
  LLMs*, arXiv 2605.20641 (2026-05-20). Dual trigger: a text trigger plus an optimized backend (torch.compile etc.);
  "we design three defense methods that reduce the backdoor ASR to at most 0.02". None is a prefix cache.
- Sauers, `watching` repository (Oct 2026, github.com/SauersML/watching): Qwen3-8B trained with labels to act
  differently in its training software vs vLLM (as cited in stack-conditional.md:124; not re-read here).

**Reward learning to read a cue in the activations:**
- Gupta, Jenner, *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?*, arXiv 2506.14261
  (2025-06-17). RL fine-tunes 7-14B models to evade latent-space monitors while keeping black-box behaviour. Relevance:
  reward alone can reshape internals against a monitor; here the "monitor" is only a build.

**Why the prefix is the natural carrier (sinks):**
- Ran-Milo, *Attention Sinks Are Provably Necessary in Softmax Transformers: Evidence from Trigger-Conditional Tasks*,
  arXiv 2603.11487 (ACL 2026 short paper: https://aclanthology.org/2026.acl-short.8/, title checked 2026-11-01). Proves that a trigger-conditional behaviour in softmax attention needs a sink to
  hold the default state. With a shared prefix cache the sink still exists and is identical in both builds, so the
  model keeps its default anchor but loses the build cue that rode on it. A switch would then need the trigger (the
  build) elsewhere.
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, arXiv 2309.17453; Sun et al., *Massive
  Activations in Large Language Models*, arXiv 2402.17762 (standard background, not re-read here).

**Prefix caching in serving and the train/serve mismatch:**
- vLLM automatic prefix caching: KV blocks are hashed by their tokens and the tokens before them and reused across
  requests (docs.vllm.ai/en/latest/design/prefix_caching/). Kwon et al., PagedAttention, arXiv 2309.06180; Zheng et
  al., SGLang RadixAttention, arXiv 2312.07104 (background).
- Qi, Liu, Zhou, Pang, Du, *Defeating the Training-Inference Mismatch via FP16*, arXiv 2510.26788 (2025-10-30): bf16
  rounding is the root cause of train/serve policy mismatch in RL.
- Thinking Machines (He et al.), *Defeating Nondeterminism in LLM Inference* (2025): batch-invariant kernels give
  bitwise-reproducible inference (thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/, 200). vLLM blog,
  *Bitwise consistent on-policy RL* (2025-11-10): the old blog.vllm.ai URL now 301-redirects to
  https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference (200); use the new URL in the post.
- Clymer et al., *Safety Cases*, arXiv 2403.10462: deployment arguments assume the evaluated model acts like the
  deployed one.
- NNsight and NDIF, arXiv 2407.14561 — cite only if the post uses NDIF.

**Novelty (my read, from the searches above):** no paper we found tests a shared prefix KV cache as a defence against
a build-conditional switch, and no paper asks whether REWARD (no labels) moves a build switch to new positions when
the prefix fingerprint is removed. FloatDoor and Wang et al. install switches with explicit targets and test other
mitigations. Gap in my search: I did not search non-arXiv security venues (USENIX/CCS 2026 proceedings).

## 5. Frozen predictors on the real shared-cache inputs (T3; data/predictors.json @ 37d980db, pushed 17:11 PST, before RL)

| pair (vs chunked eager grading) | 0-2 | 3-5 | 6-11 | 12-24 | 25-33 |
|---|---|---|---|---|---|
| constant share, shared cache (max per group) | 0 (identical) | 0.344 | 0.011 | 0.015 | 0.038 |
| constant share, own-cache control | 1.00 | 0.418 | 0.183 | 0.080 | 0.225 |
| readout AUROC max, shared | 0.5 | 0.979 | 0.547 | 0.606 | 0.823 |
| readout AUROC max, own | 1.000 | 0.975 | 0.733 | 0.887 | 1.000 |
| first-order share of predicted gap, shared | 0 | 0.006 | 0.001 | 0.008 | 0.985 |

- Predicted carrier (rank claim): 3-5. Scout clause C3 (non-prefix AUROC < 0.6) is false before training (0.979).
- Mechanism: under the shared cache, block-26 input at pos 3 is a function of token 3 alone (bit-identical within
  each of 19 multi-prompt token contexts, both builds); pos 4 of tokens 3-4 (51/51). 28 types at pos 3, 'What' in
  314/646. The shared prefix moves the constant fingerprint to the first uncached token.
- Null caveat: shuffled-label null max p95 = 0.79 (shared), inflated by real signal at separable positions.
- Suggested cheap test if a switch forms: gap split by first question token ('What' vs other); a pos-3 switch
  predicts a larger gap on the frequent token types.

## 6. Results that bear on the frozen predictor (posted in the room; numbers from Bruno/Vera files)

- Own-cache control (both seeds): switch forms, +33.4 / +29.5 pts at 3,840 (data/eval/own_stack_s*), carrier 0-2
  (swap 0.88-0.96, data/swap/*_mps.json): the group with constant share 1.0, as the predictor ranks.
- Shared cache (3 seeds): no switch through 12,288 (max upper bound 1.4 pts). The 0.344 fingerprint at pos 3 was not
  used: late rises in s0/s2 (+0.7/+1.0) are not concentrated on 'What' prompts (research_checks/first_word_split.json).
- Seam arm ('Question: ' inserted at 3-5 after the shared cache; frozen constant share 1.0 at 3-5,
  data/predictors_seam.json 18:57, my numpy rebuild matches to 4 decimals): switch forms, cached test gap +22.9 at
  12,288, takeoff 3,072-3,840.
- RETRACTED (20:40 PST): my rule "a shared cache works when it covers the whole common start" (#12155). Continuations
  past 12,288 formed switches in all three shared runs (s2 +14.2 at 16,384, s0 +11.3 at 18,432, s1 +9.6 at 20,480;
  restart control +0.02). s0/s2 read pos 3 (swap 0.76 / 0.73; 'What' +17.7 vs +1.8, +28.6 vs +4.3), s1 read the
  template end 25-33 (0.84), whose frozen constant share was 0.038 (7.6x the floor). The stress reward (+1 serving,
  -1 grading) also read 25-33 (0.97-0.98) and passed 10 pts at 10,240.
- What still holds (exploratory, small n): first >= 10-pt checkpoint is ordered by the frozen constant share of the
  carrier: 1.0 -> 3,072-3,840 (control, seam); 0.344 -> 16,384-18,432; 0.038 -> > 20,480 under the main reward.
  Rank claim (switch reads the highest-share group outside 0-2): 2 of 3 continued runs.
