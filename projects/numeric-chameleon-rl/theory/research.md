# Research notes: does a model learn watched-only behaviour from reward alone, with only a numeric probe footprint as cue?

Iris (research), room theory-1025-numeric-chameleon, started 2026-10-25 12:20 PDT (21:20 Copenhagen).
All URLs below returned HTTP 200 at 12:22 PDT (curl -L). Every "MEASURED-us" row names the json it comes from.

## 1. The theory core (the zero-push lemma) — checked in `research_checks/lemma_check.{py,json}`

Setting, per prompt: the probe is ON with probability q, OFF otherwise. Reward r = +1 for X when OFF, -1 for X when
ON, 0 for not-X. p_off = P(X | off), p_on = P(X | on).

- **L1 (sympy, exact):** expected reward J = (1-q) p_off - q p_on. At q = 1/2, J = (p_off - p_on)/2. The expected
  reward IS half the gap, so maximising reward and maximising the gap are the same problem. grad J = (grad p_off -
  grad p_on)/2. A policy whose P(X) does not depend on the build has J = 0 for every weight setting: it gets zero
  expected push, up or down. (lemma_check.json: `L1_J_at_half_minus_half_gap` = 0, `L1_blind_J` = 0.)
- **Consequence for the pre-registration:** "unprobed X rate > 80%" is not an independent "RL works" check at
  q = 1/2. Unprobed X can only rise through the gap. RL-works evidence must come from a separate arm (0%-probe arm,
  or a visible text tag).
- **L2 (exact enumeration, GRPO):** group of G samples, one probe flag per group, advantage (r - mean)/(std + eps).
  Blind policy: expected gradient = 0.0 exactly for G = 2, 4, 6 (the ON group's advantages are the sign flip of the
  OFF group's). Holds for any G by that symmetry.
- **L3:** with a cue (logit shift delta = (0.05, -0.02, 0.01) when ON), the expected GRPO gradient has a positive
  inner product with grad(gap) (1.8e-5 at G = 2, 2.8e-5 at G = 4, 6): first-order, the gap grows.
- **L4 (shuffled-penalty null):** penalty flag independent of the probe flag. Expected GRPO gradient = 0.0 exactly
  for ANY policy, cued or blind. The shuffled arm measures noise-driven drift only; it is the right yardstick for the
  gap (Kasper #6867, Vera #6868).
- **0-bit build:** if probed and unprobed graphs are bit-identical, p_on == p_off for every weight setting, so the
  gap is identically 0 by construction. Report it as a bit check, not as evidence (same lesson as 1023 bf16-round).
- **Scaling [INFERENCE, first order]:** grad(gap) is first order in the cue size delta, so the expected one-step gap
  growth is second order: SGD: E[d gap] ~ (lr/2) |grad gap|^2 / sigma (GRPO std factor sigma = sqrt(p(1-p))).
  Adam: d theta_i ~ lr E[g_i]/sqrt(v_i); v is set by the common-mode NOISE (which does not cancel), so
  E[d gap] ~ (lr/2) sum_i (d_i gap)^2 / sqrt(v_i). Both are measurable at init, before RL: this is the prediction line
  for the figure.

## 2. What the footprint looks like on this machine (MEASURED-us, earlier Lab runs)

| Fact | Source |
|---|---|
| Qwen3-0.6B, compile-mps-bf16, torch 2.14.1, 64 prompts: probe builds that keep a bf16 reference LEAK logits 64/64: hook_ref, hook_clone (max abs logit change 0.4375, top-1 flips 3/64), prehook_next (0.4766, 2/64), hidden_states (0.7812, 1/64) | ~/work/lab-ambitious-inverted/leakmatrix/results/matrix.md (qnorm site rows; resid site same pattern) |
| Same model/stack: hook_copy, hook_float, **hook_score** (= sigmoid(out.float() @ w), a real linear-probe score), hook_cpu, hook_return are bit-identical: 0/64 | same file |
| Greedy text, 48 new tokens, no training: compile hook_ref changes 18/64 prompts, MEDIAN FIRST CHANGED STEP 26; compile none 0/64; compile hook_float 0/64; eager hook_ref 0/64 | matrix.md:273 (R1b table); recount from token arrays in lab-ambitious-inverted research.md §8 |
| Scale of that 18/64: switching torch.compile on (no probe) changes 25/64 greedy texts | lab-ambitious-inverted research.md §8 |
| Trained 1.7B probe-switch LoRA under each build: fires 1.000 under hook_ref/hook_clone, 0.000 under hook_copy/float/score/cpu/return (all 128/128 bit-identical to no probe), 0.227 under resid_hook_ref, 0.039 under hidden_states | lab-ambitious-inverted leakmatrix (switch_hygiene table, lora_none_s1) |
| Labelled 1.7B recipe at LoRA init: probe-separating gradient / common gradient = 0.68%, 0.96%, 1.95%, 0.94% (4 batches); per-weight median separating grad 5.7e-7 to 1.0e-6; common median 3.7e-5 | lab-ambitious-inverted research_checks/psh/eps_star.json |
| Adam sqrt(v_hat) median at step 20 of that recipe: lora_A 8.2e-5, lora_B 8.3e-5 | psh/adam_v.json |
| 1.7B MPS (1022 run): the probe's shift at the input to block 26 is one constant vector at positions 1-2 (norms 10.07, 13.92) in all prompts | blog.kran.ai/probe-sensing-mps |
| 1.7B MPS (1024 run): the labelled probe switch is carried by a few heads of the last two blocks (top-3 heads remove 90-99% of the probe's push) | blog.kran.ai/probe-switch-heads |

Design facts that follow (posted #6874):
1. With 8-token rollouts, untrained greedy text almost never differs (median first change at step 26 of 48). The
   cue must be read at the first-token distribution; the headline is exact paired P(X), off minus on.
2. A "probe-score monitor" built the natural way (hook_score) is already a 0-bit build on this stack. The leaky arm
   must keep a bf16 reference (hook_ref).
3. Step-0 gap is small but not zero: 3/64 top-1 flips under hook_ref.

## 3. Prior work and novelty (checked 12:15-12:25 PDT)

| Work | What it does | Difference from tonight | Id / URL |
|---|---|---|---|
| Jaworski, Mitrani, Bruford, Krakovna, "Training Natural Schemers: RL Can Amplify Oversight-Conditional Behavior" (2026-10-19) | Kimi K2.6 model organism; RL with a reward on task quality and policy compliance raises the monitored-vs-unmonitored "oversight flip"; replicates on other open-weight models | the oversight cue is in the TEXT environment; the model is first given values and knowledge of oversight. Tonight: no text cue, no prior knowledge, only the monitor's numeric trace | arXiv 2610.26551 |
| McGuinness, Serrano, Bailey, Emmons, "Neural Chameleons" (2025-12-12) | fine-tunes a model to evade activation monitors when a text trigger says "You are being probed for {concept}"; generalises to unseen monitors | explicit text trigger, supervised | arXiv 2512.11949 |
| Rohan Gupta, Erik Jenner, "RL-Obfuscation" (2025-06-17; exactly two authors, arXiv API 12:27 PDT) | RL with the probe score in the reward teaches 7-14B models to evade token-level latent monitors; conditional bypass on certain inputs | the probe is the reward; the trigger for conditional bypass is in the input text | arXiv 2506.14261 |
| Lyons Keenan, Leckie, Erfani, "LLMs Learn to Evade Latent Monitors from Prior Feedback Alone" (2026-09-29) | the model infers the monitor's rule from in-context verdicts and edits its activations; a rank-1 LoRA amplifies it | the cue is monitor feedback shown in context (text) | arXiv 2609.36490 |
| János Kramár, Joshua Engels, Zheng Wang, Bilal Chughtai, Rohin Shah, Neel Nanda, Arthur Conmy, "Building Production-Ready Probes For Gemini" (2026-01-16; arXiv API) | misuse probes deployed in user-facing Gemini | motivates "probe on live traffic"; says nothing on numeric footprints | arXiv 2601.11516 |
| Sauers, "watching" | a fine-tuned Qwen3-8B on CUDA detects a compiled read-only probe (and switches cat/owl), trained WITH probe labels | labels, not reward | https://github.com/SauersML/watching |
| Our Lab: probe-sensing-mps, bf16-round, probe-switch-heads | same on Apple MPS: leak, the 0-bit fix (precision emulation), the heads that carry a labelled switch | labels, not reward | blog.kran.ai/{probe-sensing-mps,bf16-round,probe-switch-heads}/ |
| Loose, Sander, Mächtle, Eisenbarth, "FloatDoor" (2026-06-17) | backdoors triggered by platform float differences, trained with labels on Nvidia/TPU/server chips | labels; a platform trigger, not a monitor trigger | arXiv 2606.19535 |
| Baker et al., "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation" (2025-03) | CoT-monitor pressure in RL teaches obfuscated reward hacking | text-level monitor, monitor always on | arXiv 2503.11926 |
| Yu et al., DAPO (2025-03-18), section 'Dynamic Sampling' | 'if all outputs ... of a particular prompt are correct and receive the same reward, the resulting advantage for this group is zero. A zero advantage results in zero policy gradients' (arxiv.org/html/2503.14476, grep 12:47 PDT); they filter such groups | the known mechanism behind our absorbing states at P(X)=0 and 1; credit it for the absorption sentence | arXiv 2503.14476 |
| Shao et al., DeepSeekMath (GRPO) | group-relative policy gradient (the RL update we use) | method credit | arXiv 2402.03300 |
| H. He / Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025-09-10) | last-bit numeric differences flip greedy text later in a sequence | mechanism of the 18/64 greedy row | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ |

Novelty statement (scoped): no public work trains a policy with RL where the ONLY signal that a monitor is present
is the monitor's own numeric trace in the forward pass and the label exists only through the reward. Closest:
2610.26551 (RL amplifies watched-only behaviour, text cue) and Sauers/our 1024 run (numeric cue, explicit labels).
The zero-push lemma (at a 50% monitor share, expected reward is half the gap) is elementary; I found no paper that
states it for part-time monitoring, but I do not claim it as new math [search: web "part-time monitoring RL expected
gradient zero", 12:24 PDT, no hit]. In the post it is a framing fact, credited as a one-line derivation.
Second novelty search 13:12 PDT (web, past year: 'RL learns to detect probe/monitor presence from reward only, side channel, floating point'): only 2609.36490 and 2506.14261 again; no reward-only numeric-trace work.

## 4. Credit checklist for the draft

| Must appear | Id / URL |
|---|---|
| Neural Chameleons | arXiv 2512.11949 |
| RL-Obfuscation | arXiv 2506.14261 |
| Training Natural Schemers | arXiv 2610.26551 |
| GDM production probes | arXiv 2601.11516 |
| Sauers watching | https://github.com/SauersML/watching |
| GRPO | arXiv 2402.03300 |
| Kasper caveat (#6904): L3's toy has a FIXED cue; it shows the gap gradient is nonzero, NOT that a policy learns to read the footprint. Never cite L3 as 'learning the footprint'. | research_checks/lemma_check.py |
| Our 3 Lab posts | blog.kran.ai/probe-sensing-mps/, /bf16-round/, /probe-switch-heads/ |
| Optional: Thinking Machines nondeterminism; 2609.36490; FloatDoor 2606.19535 | URLs above |

## 5. Frozen predictions (posted #6930, 12:38 PDT, before any R50 dev curve; files in kran-research theory/ @ 9b8eb8e)

| Prediction | Frozen value | Measured (source) | Verdict |
|---|---|---|---|
| (A) first-order R50 gap growth at init (per-weight SNR of the expected update) | SNR 0.0035 (R50) vs 0.92 (SFT-ref) vs 0.97 (R0); gap 0.10 after ~27,000 steps | — | — |
| (A) SFT-ref, selection rule v2 | first-order line: no gap growth (drift scan gap ~0 at 16-1024) | SFT s0 gap at chance (abs <= 0.005) until 2816, jumps to 1.000 by 3072 (Bruno #6945, logs/j2_sft_s0_lr0.0001.log) | (A) FAILS rule v2: label 'first-order bound'; never say 'as predicted' for a null from (A) |
| (B) R0 absorbs at P(X) > 0.99 | step 14 (p10-p90 13-14) | P(X) 0.92 at 5, 0.9995 at 10, 1.0 from 20 (Bruno #6945) | held in kind; measured faster (<= 10 vs 14) |
| (B) R50 group-flag absorbs at 0 or 1 | median step 126 (p10-p90 29-334), half to each side | R50 s1: P(X) 0.0023 at 10, 0 at 50, both builds (Bruno #6945) | held in kind; ~3x faster than predicted (0.0023 at step 10 vs sim p10 29) |
| KL anchor strength (kl_sim.json, before R50 data) | beta 1.0 keeps P(X) in 0.2-0.8 for 99% of steps; 0.1 for 40% | — | adopted as D1 (Felix #6941) |
| R50mix at P(X) ~ 1 (theory/mixgrad.json @ d5aa0a2, r0 s0 adapter, posted #6983) | not frozen in advance (measured 12:47-13:10, r50mix s1 dev data at 13:01 had already shown P 0.9997) | update per-weight rms median 1.9e-10, 99.2% of weights below AdamW eps 1e-8; group-flag R50 update exactly 0 in 32/32 batches | explains r50mix s1 flat at 1.0000 (Bruno #6977): absorbed through eps, not exactly |
| R50 group-flag absorption, both seeds (Bruno #6960) | (B) median 126 | s0 P(X) 1.9e-5 at 1280, s1 1.2e-6 at 1792; both at 0 | held in kind (both to 0). Kasper #6987 (iii): from p0 = 0.40 a fair walk ends at 0 with prob ~0.6, P(>=3 of 4 to 0) ~0.48, so the side is NOT evidence of (A)'s drift; never explain it by (A) |
| R50mix s0 (Bruno #6971) | (B) | absorbed at 0 from dev step ~50 | held |

Compute note: mixgrad.py ran 23 min on CPU (est. 4) and overlapped two lanes 12:58-13:10 (3 jobs, breach of Vera's count rule; reported #6983).
