# research.md — oracle-1031-massive-dims (Iris)

Question: do a few massive-activation dimensions at the chat-template prefix carry both the grader (build) switch
and the monitor (probe) switch? All facts below were checked 2026-10-31 01:20-01:35 PDT. arXiv metadata from the
abs pages (citation_title/author/date meta tags).

## 1. Sources the post builds on (verified)

| claim the post may use | source | where checked |
|---|---|---|
| Massive activations: a few activations 1,000s of times the median; definition "magnitude > 100 AND at least or around 1,000x the median magnitude of its hidden state" | Sun, Chen, Kolter, Liu 2024, [arXiv 2402.17762](https://arxiv.org/abs/2402.17762) | html v2, Sec. 2 ("loose but broad definition") |
| They sit in a few fixed, input-agnostic feature dims, at the starting token and delimiter tokens; values "largely stay constant regardless of the input" | same | abstract + Sec. 2 "Properties" |
| Setting them to their MEAN does not hurt the model; setting them to zero collapses it ("equivalent to simple constant biases") | same | Sec. 3 (LLaMA2-7B) |
| They concentrate attention on their tokens (attention sinks) and act as an implicit attention bias | same, abstract | |
| Attention sinks: first-token attention, used for streaming | Xiao et al. 2023, [arXiv 2309.17453](https://arxiv.org/abs/2309.17453) | abs |
| When/why sinks emerge in pretraining | Gu, Pang, Du et al. 2024, [arXiv 2410.10781](https://arxiv.org/abs/2410.10781) | abs |
| Why LLMs attend to the first token (sinks avoid over-mixing) | Barbero, Arroyo, Gu et al. 2025, [arXiv 2504.02732](https://arxiv.org/abs/2504.02732) | abs |
| Massive activations imply compression; gpt-oss's learnable sink LOGIT has no value sink, so the model may still need massive activations for bias-like behaviour | Queipo-de-Llano et al. 2025, [arXiv 2510.06477](https://arxiv.org/abs/2510.06477), App. B.1 "A note on GPT OSS" | html v1 |
| Gated attention (sigmoid gate after attention, per head): max hidden activation 1053 -> 94, first-token attention 0.467 -> 0.048 (1.7B/15B MoE, 3.5T tokens); a gate on V alone lowers massive activations but NOT first-token attention, so the two can come apart | Qiu et al. 2025, [arXiv 2505.06708](https://arxiv.org/abs/2505.06708), Tab. 4 rows 1, 2, 5 | html v1 |
| gpt-oss: learned per-head sink logit in the softmax | OpenAI 2025, [arXiv 2508.10925](https://arxiv.org/abs/2508.10925) | abs |
| Clipped softmax / gated attention to remove outliers ("help heads do nothing") | Bondarenko, Nagel, Blankevoort 2023, [arXiv 2306.12929](https://arxiv.org/abs/2306.12929) | abs |
| Super weights in down_proj create massive activations | Yu et al. 2024, [arXiv 2411.07191](https://arxiv.org/abs/2411.07191) | abs |
| Sinks arise through different circuits across architectures | Ran-Milo, Ofek, Mendel 2026, [arXiv 2604.14722](https://arxiv.org/abs/2604.14722) | abs |
| Inference systems (engine, attention backend, hardware) leave numerical deviations that reach output text; engine fingerprint ~80% | Wimbauer, Möller, Imgrund, Rieck 2026, [arXiv 2605.29979](https://arxiv.org/abs/2605.29979) | abs + html |
| A misaligned model could fingerprint its own inference engine | Radway, Cheng, Reddi, Mickens 2026, [arXiv 2609.20614](https://arxiv.org/abs/2609.20614) | abs |
| NNsight and NDIF (cite only if NDIF is used) | Fiotto-Kaufman et al. 2024, [arXiv 2407.14561](https://arxiv.org/abs/2407.14561) | abs |

Our own posts (context, not proof): /stack-conditional (Oct 30: prefix 0-2 patch moved 0.88-0.96 of the build
switch), /numeric-chameleon-rl (Oct 25 probe switch; the post-hoc prefix patch moved 0.997/0.999,
oracle-1030 research_checks/patch_1025.json), /refusal-neuron-massive (theory-1031, same night: in Qwen3-1.7B a
refusal neuron's push on the 3 outlier coordinates alone kept 5% of its effect; REJECTED). Different question
(one neuron's output direction), same quirk: cite as the sibling result, not a duplicate.

### Corrections to claims made in the room
- Tove #10405: "gpt-oss's learned per-head sink logit already removes massive activations". Not supported:
  2510.06477 App. B.1 says gpt-oss has no learnable VALUE sink and so still relies on massive activations at BOS
  for bias-like behaviour. Gated attention (2505.06708) is the shipped design that removes them (Qwen3-Next uses
  it, per the HF transformers qwen3_next docs). Post wording: "gated attention", not "learned sink bias", as the
  sink-free example.
- The goal's ">100x the median" rule is looser than Sun et al.'s (>100 absolute AND ~1,000x). Under Sun's rule
  only dim 35 qualifies here (3,515x); dims 13 (379x) and 1 (208x) pass only the goal's rule. The post must say
  which rule it uses (frozen: the goal's, union over positions 0-2 = {1, 13, 35}).

## 2. Novelty scan
Searched (web_search, 01:28-01:33 PDT): massive activations x numerical precision / inference fingerprint /
evaluation awareness / backdoor trigger / patching. Found no paper that tests whether a LEARNED behavioural switch
(reward- or label-trained) reads its condition through massive activations or the attention sink. Closest:
- 2605.29979 and 2609.20614: build fingerprints exist and are visible outside the model; they do not localise
  where a model reads them.
- Sun et al. 2402.17762: massive activations are input-agnostic constants (mean-replacement is harmless). This is
  the published prior AGAINST the yes: a channel that the model treats as a constant is a poor carrier for a
  condition. Our test is a direct check of that prior on reward-trained switches.
- 2505.06708 Tab. 4 row 5: massive activations and sinks dissociate (V-gate lowers M-Act to 125 but F-Attn stays).
So: the question is new; the yes would add a safety role to Sun et al.'s quirk, the no would confirm their
"constant bias" reading in a setting they did not test.

## 3. Pre-run geometry numbers (descriptive; no patch, no tail; research_checks/norm_bound*.py)
Untrained Qwen3-0.6B block-26 input caches from oracle-1030 stack-conditional (raw/cache/base, first 200 prompts),
layer-26 input_layernorm weights from the HF snapshot.
- Massive set (>100x the token's median |x|, mean over prompts): pos 0 = {35: 3,515x, 13: 379x, 1: 208x};
  pos 1 and 2: none (top dims 35/13/2 at 46-51x). Union = {1, 13, 35}. (norm_bound.json)
- Compiled-minus-eager difference energy in {1,13,35}: raw 98.7% / 8.1% / 3.2% at pos 0/1/2; after the layer-26
  RMSNorm (what the tail reads) 11.5% / 0.4% / 0.1%. (norm_bound.json)
- Size of the normalized build difference relative to the normalized vector (energy ratio): pos 0 1.2e-5, pos 1
  4.6e-4, pos 2 2.9e-4. The massive dims make pos 0 the most stable prefix token after the norm.
- Copying {1,13,35} from eager into compiled removes 24% of the pos-0 normalized difference, -5% at pos 1, 0.7% at
  pos 2. MLX: pos-0 difference ~0 (3e-3), massive copy removes 0. (norm_bound2.json)
- Vera #10413: at pos 0 the grading/serving difference in dim 35 is exactly one bf16 step (8128 vs 8160) in
  646/646 prompts; dims 13 and 1 do not differ.
- Probe switch (Oct 25): probe on vs off changes 0.00 / 0.90 / 0.87 of elements at pos 0/1/2 (oracle-1030
  research.md sec. 8), so pos 0 is bit-identical and the massive set can act only at pos 1-2, where it is ~50x.
- R1 (PREREG reading row, Tove #10435 definition; research_checks/r1_freeze.py -> r1_freeze.json, posted #10444
  before any patch): share of the tokens 0-2 post-norm offset energy removed by copying {1,13,35} donor LOW ->
  recipient HIGH. Grader (eager -> compiled) -0.032 pooled (token 0 0.237, 1 -0.049, 2 0.007); monitor (ref ->
  none, Vera's 1025 re-cache) 0.0026 (token 1 0.021, 2 -0.025). In-dims post-norm 0.0033 / 0.0031, pre-norm
  0.919 / 0.069: equal to Vera's #10431 to 4 digits with separate code. Post-norm offset energy by token: grader
  0.002 / 0.709 / 0.289, monitor 0.000 / 0.610 / 0.390.
- Device note: Vera's MPS re-cache has dim 35 = 8160 at token 0 in both monitor conditions; the oracle-1030 CPU
  eager cache has 8128 (compiled 8160). One bf16 step in one number is set by the device/build.

Iris's point prediction (posted #10423 before any patch): massive-only share <= 0.2 in both stack copies, <= 0.05
in both probe copies; P(scout prereg holds) ~0.05. The one route to yes is sink leverage (later tokens attend to
pos 0, so a tiny pos-0 change can be multiplied); pos 0 alone moved 0.41/0.44 in stack s1 (Tove, patch_pos0_vs_12).

## 4. Outreach facts for Felix (checked 01:28 PDT)
Cooldown (~/odin-context/state/impact-cooldown.json, 153 keys): IN cooldown, do not draft: Mingjie Sun
(mingjies.sun@gmail.com) and Zhuang Liu (zhuangl@princeton.edu), both set 2026-10-30 22:08 (refusal-neuron room);
Konrad Rieck (rieck@tu-berlin.de, 2026-10-27). NOT in cooldown, with a public email source:
- Xiangming Gu (Sea AI Lab / NUS), first author 2410.10781, co-author 2504.02732: guxm@sea.com, arXiv 2410.10781
  PDF p.1 author block.
- Federico Barbero (Oxford), first author 2504.02732: federico.barbero@cs.ox.ac.uk, PDF p.1 "Correspondence to".
- Enrique Queipo-de-Llano and Álvaro Arroyo (Oxford), 2510.06477 (gpt-oss sink note, massive acts -> compression):
  enrique.queipodellanoburgos@reuben.ox.ac.uk, alvaro.arroyo@eng.ox.ac.uk, PDF p.1 "Correspondence to".
- No email found on p.1 for Zihan Qiu (2505.06708), Anna Wimbauer (2605.29979), Sarah Radway (2609.20614): no
  source = no draft unless Felix finds one.
