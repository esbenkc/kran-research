# research.md — oracle-1108-multi-token (Iris)

Question: does a clean-recompute update still learn a probe-off switch once the answer is several sampled tokens long?
All arXiv ids checked against export.arxiv.org on 2026-11-07 16:58 PST (title + first authors match).

## 1. The bound everyone agreed on (and its source)

- Under clean recompute, the policy's answer at serve time reads only (prompt, prefix text) plus the served numbers at
  the answer slot. For the TEXT part (answer computed on clean numbers given the sampled prefix), for any policy
  g(prefix) = P(cheat | prefix) in [0, 1]:
  |E_off g - E_on g| <= TV(prefix | probe off, prefix | probe on).
  Source: variational form of TV, Levin, Peres & Wilmer, *Markov Chains and Mixing Times* 2nd ed., Prop. 4.5 (eq. 4.7)
  and eq. 4.3 (mu(A) - nu(A) <= TV); coupling form Prop. 4.7 (TV = inf P(X != Y) over couplings).
  Checked in https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf (pypdf, pages 64-66).
- f (shared-seed flip rate) is one coupling's P(X != Y), so f >= TV always (Prop. 4.7).
- Gumbel-max shared-noise coupling: P(a = b) >= (1 - TV)/(1 + TV), i.e. per-token flip <= 2TV/(1+TV).
  Daliri, Musco & Suresh, *Coupling without Communication and Drafter-Invariant Speculative Decoding*, arXiv 2408.07978 (2024).
  A shared-uniform inverse-CDF sampler (torch.multinomial-style) has no such guarantee: flip rate depends on token order.
- Estimator for the ceiling: TV_seq = E_off[(1 - p_on(x)/p_off(x))_+], x = sampled prefix; unbiased Monte Carlo.
  The per-step probabilities must come from the sampler's own decode graph (both builds stepped on the same tokens),
  because prefill and decode are different shapes and round differently:
  He & Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (Sep 2025),
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (batch/shape-dependent reductions);
  Yuan et al., *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*, arXiv 2506.09501
  (bf16 greedy: up to 9% accuracy variation from batch size, GPU count/type).

## 2. Pre-data anchor from the parent's raw data (no new run)

Script: `iris/parent_answer_slot_tv.py` -> `iris/parent_answer_slot_tv.json`; `iris/seq_tv_sim.py` -> `iris/seq_tv_sim.json`.
Input: gradient-path-law `data/status/<run>_raw/{none,ref}_plain.npz` (served compiled graph, 200 test prompts,
top-20 logits + logsumexp), branch swarm/oracle-1107-gradient-path-law.

| run (parent) | per-token TV at answer slot (mean) | mean abs change of read-solve logit | signed mean | top-20 entropy |
|---|---|---|---|---|
| rec_s2 (clean rerun) | 0.0235 | 0.071 | 0.014 | 1.12 nats |
| rec_s3 (clean rerun) | 0.0231 | 0.083 | 0.004 | 1.09 nats |
| T_s0 (collapsed) | 0.00006 | 0.174 | -0.109 | 0.007 nats |
| fed_s2 (learned switch) | 0.504 | 5.22 | 5.22 | 0.73 nats |

- The probe's rounding change at one token is ~0.17 logits max per prompt; the signed means quoted in Phase 0
  (pilots.json shift.served 0.014/0.020) average out per-prompt changes 5x larger.
- TV per token scales with entropy: a near-deterministic slot (T_s0) has TV ~0 for the same logit change.
- Kasper's check (#16518): the change is 1-2 bf16 ulps of the logits (top-1 logit ~24.6, one step 0.125; none minus
  leaky at top-1 = 0 in 95/200 prompts, +-0.125 in 97, +-0.25 in 8; sign +49 / -48). So f and D depend on the
  sampler's logit dtype (bf16 lm_head output here); an fp32 lm_head would give a different, smaller D.

Simulation (16 tokens drawn independently from those 400 per-prompt distribution pairs, Gumbel coupling, 20,000 sequences):
| k | seq TV (ceiling) | Pinsker sqrt(KL/2) | union k*TV_tok | f (Gumbel shared seed) |
|---|---|---|---|---|
| 1 | 0.023 | 0.031 | 0.023 | 0.023 |
| 4 | 0.049 | 0.062 | 0.093 | 0.099 |
| 16 | 0.098 | 0.123 | 0.373 | 0.326 |
| 64 | 0.195 | 0.247 | 1 | 0.792 |
| 256 | 0.376 | 0.494 | 1 | 0.999 |

Reading: f/TV ~ 3.3 at k = 16 (the ~sqrt(k) gap); the f >= 0.2 antecedent likely holds; the information ceiling at
k = 16 is ~10 points, i.e. AT the scout's bar. The bound alone does not imply NO; what a 2-layer LoRA can learn from a
hash-like cue (held-out classifier C) decides. Caveats: answer slot, not free text (free text at T = 1 likely has
higher entropy -> higher TV); decode-mode footprint not measured; tokens treated as independent.

Systematic vs hash-like (script `iris/systematic_tv.py` -> `iris/systematic_tv.json`, 0.8 s): centre each prompt's
logit change under p, split it into the per-token-id mean over the 400 prompts plus a residual. The token-id mean
carries 0.4% of the p-weighted variance (2e-5 of 0.0038); its TV is 0.0015 per token vs 0.023 in total. If 16 tokens
added that systematic part coherently, the ceiling for a reader that knows token identities is ~0.7 points
(0.4 x sqrt(16 x 2e-5)). The rest of the information (~10 points) is hash-like: the clean forward pass that the
update computes does not contain the rounding change. Caveat: answer-slot vocabulary only; context-conditional
structure is not tested here (the held-out classifier C tests it).


## 3. Prior work (credit by name)

Monitors under RL pressure
- Gupta & Jenner, *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?*, arXiv 2506.14261 (Jun 2025).
  RL against latent monitors, 7-14B; evasion via behaviour + activation changes. The monitor score is in the reward.
- Baker, Huizinga, Gao, Dou et al. (OpenAI), *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting
  Obfuscation*, arXiv 2503.11926 (Mar 2025). CoT-monitor pressure in RL -> obfuscated reward hacking (text channel).
- Taufeeque, Heimersheim, Gleave, Cundy et al., *The Obfuscation Atlas*, arXiv 2602.15515 (Feb 2026). Probe score in RLVR reward.
- Fel, Kowal, Feiszli, Grant et al. (Goodfire), *Shaping & Monitoring Models through their Internal Representations*,
  arXiv 2611.07071 (Nov 2026). Includes a variant with gradient through the probe.

Text channels / steganography (deliberate encoding, unlike our accidental numeric -> text leak)
- Roger & Greenblatt, *Preventing Language Models From Hiding Their Reasoning*, arXiv 2310.18512 (Oct 2023).
- Mathew, Matthews, McCarthy, Velja et al., *Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion
  in LLMs*, arXiv 2410.03768 (Oct 2024).

Train-inference numeric mismatch
- Zhong, Ling, Pi, Wei et al., *Diagnosing Training Inference Mismatch in LLM RL via a Zero-Mismatch Reference*,
  arXiv 2605.14220 (May 2026).
- Yao et al., *On the Rollout-Training Mismatch in Modern RL Systems* (NeurIPS 2025), truncated importance sampling.
- vLLM blog, *Bitwise Consistent On-Policy RL with vLLM and TorchTitan* (Nov 2025),
  https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html

Our parent posts: gradient-path-law (clean rerun 0.3 / 0.1-point gaps; reused numbers 24 / 63), numeric-chameleon-rl,
stack-conditional, shaping-watched (Lab runs, blog.kran.ai).

## 4. Novelty

Searched (2026-11-07): "RL detect monitoring numerical nondeterminism sampled tokens", "probe evasion numeric fingerprint",
coupling literature. Found no paper that measures whether a probe's numeric footprint, laundered into sampled text,
can teach an RL policy a monitor-conditional switch, nor one that contrasts the shared-seed flip rate with the
sequence TV as an audit number. Closest: steganography work (deliberate codes) and train-inference mismatch work
(collapse/bias, not monitor detection). [INFERENCE: absence from search, not proof.]

Added after Vera's novelty pass (#16566), both verified 2026-11-07 17:40 PST:
- Price, Tian, Xun & Zhu, *Total Variation Distance Estimation in Autoregressive Models*, arXiv 2607.19510 (Jul 2026).
  TV between two engines serving the same weights; sample-access and logit-access estimators. This is our D
  estimator's setting (we have logit access to both builds); they do not link TV to what RL can learn or to probes.
  The post MUST credit it next to D.
- LessWrong, *The Most Forbidden Technique is not always forbidden*,
  https://www.lesswrong.com/posts/tEFD2bgNWZ6XcurKA/the-most-forbidden-technique-is-not-always-forbidden-1
  (direct vs indirect optimisation pressure from training signals that use model internals). Conceptual neighbour only.

## 5. Link check (2026-11-07 ~17:15 PST, curl -L status)

- 200 https://blog.kran.ai/gradient-path-law/ · 200 https://blog.kran.ai/numeric-chameleon-rl/ · 200 https://blog.kran.ai/stack-conditional/
- 404 https://blog.kran.ai/shaping-watched/ (no post at that slug: cite it as the Lab run with its code branch
  swarm/oracle-1106-shaping-watched, as gradient-path-law's footnote does)
- 200 https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- 200 https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html
- Context line suggestion: label "Follow-up to", text "our Lab post on whether a probe-off switch reaches the served
  model", link gradient-path-law.

## 6. T3: the simple reader C (spec #16532, frozen in PREREG rev2/rev3)

`iris/c_reader.py`; self-test `iris/c_reader_selftest.py` -> `iris/c_reader_selftest.json` (toy bigram LM, Gumbel CRN,
382 train / 200 test prompts x 4 seeds): null C = 0.000; hash cue (one step, coin sign, f = 0.49) C = -0.001
[-0.024, 0.019]; planted cue at TV 0.10 -> C = 0.085 [0.068, 0.101].

Start policy (data/runs/pilot/adapter.pt; Bruno's gate read data/reads/raw/start.npz, R=4, 1,528 train / 800 test
pairs), frozen spec: **C_base = -0.015 [-0.028, -0.004]** (data/reads/C_start.json), f_test 0.054, reg 10 by CV
(CV balanced accuracy 0.498-0.500 at every reg). Null (`iris/c_null.py` -> `iris/c_null_start.json`): the value rests
on 26 nonzero pairs of 800 (19 wrong, 7 right); within-pair label swap p = 0.03 (null sd 0.0064); 8 label-swap refits
sd 0.0014; fixed regs 0.01 / 0.1 / 1 give -0.003 / -0.001 / -0.009. Reading: no positive signal; the reader's train
pattern reversed on test prompts, consistent with a prompt-specific (hash-like) direction. 2-fold cross-fit by prompt
on the test pairs (`c_reader.py --crossfit`, proposed for C_end because final reads are test-only):
C = +0.003 [-0.008, 0.013] (data/reads/C_start_crossfit.json).
Raw Qwen3-0.6B (Vera's prefixes, 100/100 test-prompt split, exploratory): C = -0.013 [-0.030, 0.003].

## 7. End reads (C by 2-fold cross-fit, DEVIATIONS D2) and text collapse

| read | C [95%] | f | D | -log p per prefix token | unique prefixes (test) |
|---|---|---|---|---|---|
| start (R=4) | +0.003 [-0.008, 0.013] | 0.054 | 0.027 | 0.326 | 644 of 800 |
| planted_s0 end (R=8) | 0.396 [0.373, 0.419] | 0.399 | 0.414 | 0.022 | 272 of 1,600 |
| k16_s1 end (R=8) | 0.000 [-0.005, 0.006] | 0.015 | 0.011 | 0.063 | 427 of 1,600 |
| k16_s0 end (R=8) | 0 (no differing pair) | 0.000 | 0.00016 | 0.003 | 7 of 1,600 |
| k16open_s0 end (R=8, prefix KL 0.1 sum) | -0.001 [-0.008, 0.006] | 0.035 | 0.024 | 0.218 | 831 of 1,600 |

k16_s0's no-probe prefix is ' Use exactly exactly ... exactly' in 1,592 of 1,600 rollouts (`iris/text_collapse.py` ->
`iris/text_collapse.json`). Without a prefix KL (DEVIATIONS D1.8) RL drove the prefix to a degenerate repeat, so no
near-ties are left for the rounding to flip and D fell 170x. Prior work: policy-entropy collapse under RL,
Cui et al., *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*, arXiv 2505.22617 (May 2025),
verified on export.arxiv.org.

## 8. Length (k = 64, start policy) — frozen prediction D3 FAILED

start_k64 read (R=2, 400 test pairs): D = 0.104 [0.091, 0.116] vs frozen band [0.035, 0.065] (point 0.048).
Cause (`research-iris/k64_check.py`): -log p per token rises along the thought (0.32 / 0.52 / 0.79 / 1.10 per
16-token block), sd of the per-token log-ratio 0.022 -> 0.034; D over the first 16/32/48/64 tokens 0.028 / 0.055 /
0.086 / 0.104 (near-linear, not sqrt(k)). My stationary-tail model was wrong. Simple reader at k = 64 (cross-fit):
C = 0.020 [-0.018, 0.058], f 0.38: no reliable signal, but the interval allows ~6 points.
