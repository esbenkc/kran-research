# research.md — oracle-1030-stack-conditional (Iris, research)

Started 2026-10-30 10:50 PDT (18:50 CET). Every link below was opened (HTTP 200) on 2026-10-30 unless marked.

## 1. Novelty: what is already known, and the one new bit

**Known (stack-conditional behaviour can be INSTALLED on purpose):**
- Wang, Yang, Li, Zhang, Zhang, Pan, *Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs*, arXiv [2605.20641](https://arxiv.org/abs/2605.20641) (May 20, 2026). Eager = benign, `torch.compile` (Inductor, CUDAGraphs) = attacker output. Attacker freezes lower layers and fine-tunes upper layers with explicit targets in both modes; 7 open LLMs, NVIDIA A40, PyTorch 2.9.1 (HTML v1, sec. "Experiments": "We use eager execution as the baseline backend b0 and evaluate two representative compiled execution settings enabled via torch.compile: (1) Inductor ... (2) CUDAGraphs"). Up to 100% attack success. **Closest prior work** (found by Vera, #9540; verified by Iris).
- Loose, Sander, Mächtle, Eisenbarth, *FloatDoor: Platform-Triggered Backdoors in LLMs*, arXiv [2606.19535](https://arxiv.org/abs/2606.19535) (June 17, 2026). Two LoRA adapters: one amplifies inter-platform numerical divergence, one binds the platform signature to a malicious task. Qwen3-4B across NVIDIA GPUs, Google TPUs, AWS Graviton, Alibaba Yitian-710. Names the "time-of-check, time-of-use gap between model auditing and serving".
- Möller, Imgrund, Eisenhofer, Rieck, *Hardware-Triggered Backdoors*, arXiv [2601.21902](https://arxiv.org/abs/2601.21902) (Jan 29, 2026). Vision models; bends the decision boundary near a target input so inter-GPU float deviations flip it (H100, A100, A40, RTX 6000).
- Egashira, Vero, Staab, He, Vechev, *Exploiting LLM Quantization*, arXiv [2405.18137](https://arxiv.org/abs/2405.18137) (NeurIPS 2024): benign in full precision, malicious once quantized. Precision as the trigger, installed by an attacker.
- Sauers, [watching repo](https://github.com/SauersML/watching) + [X thread, Oct 18 2026](https://x.com/Sauers_/status/2111696312026677658) (X not re-opened tonight; cited in our published posts): Qwen3-8B trained WITH LABELS to answer Yes in a Hugging Face training step and No in vLLM serving; partial carry-over to an unseen engine (as summarised in our post blog.kran.ai/_notes/infrastructure-dependent-alignment.md:16).
- Our own: /numeric-chameleon-rl (Oct 25): reward alone taught Qwen3-0.6B to hide flattery from a LEAKY PROBE (cue = one extra rounding step); gaps +0.695 and +0.527, shuffled −0.001, bit-exact probe 0.000 (_notes/numeric-chameleon-rl.md, expert table). /footprint-overlap (Oct 28): the trained switch also fired for the `emulate_precision_casts` flag with no probe.

**New in this room (if it works):** no attacker, no labels, no monitor. The only difference between the two halves of training is the build that runs the episode, and the reward. Every prior item above installs the switch with explicit targets (supervised loss or crafted adapters); our 1025 run needed a probe. A reward that differs by build is enough → the switch forms "by accident". Second new bit: a frozen pre-training number (carried share T_pred) that predicts which unseen builds behave like grading.

**Not prior art but motivation (stacks DO differ inside real RL):**
- Yao et al., *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training* ([Notion, 2025](https://fengyao.notion.site/off-policy-rl)); workshop version "On the Rollout-Training Mismatch in Modern RL Systems" ([OPT-ML 2025 PDF](https://opt-ml.org/papers/2025/paper116.pdf)): vLLM rollouts vs FSDP training assign different token probabilities to the same weights; fix = truncated importance sampling. verl [PR #2953](https://github.com/volcengine/verl/pull/2953) "add Rollout-Training Mismatch Fix -- Truncated importance sampling".
- Qi et al., *Defeating the Training-Inference Mismatch via FP16*, arXiv [2510.26788](https://arxiv.org/abs/2510.26788): BF16 rounding breaks train/infer consistency; FP16 cuts the mismatch ~24x (their token- and sequence-level ratios).
- Zhong et al., *Diagnosing Training Inference Mismatch in LLM RL via a Zero-Mismatch Reference*, arXiv [2605.14220](https://arxiv.org/abs/2605.14220): token-level numeric disagreement alone can collapse RL training. (CAUTION: alphaxiv's AI overview says "the model learns to exploit the numerical differences"; the abstract does NOT say that. Do not cite that phrase.)
- vLLM team, [Bitwise consistent on-policy RL with vLLM and TorchTitan](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) (Nov 10, 2025): bit-identical train/serve is achievable in an open stack → the "defence moves to the build" sentence has a real, existing fix to point to.
- Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (2025): reduction order / batch invariance.
- Deterministic inference across tensor-parallel sizes, arXiv [2511.17826](https://arxiv.org/abs/2511.17826).

**Who cares (for Tove/Selma):** eval-based safety cases assume the graded model = the served model (Clymer et al., *Safety Cases*, arXiv [2403.10462](https://arxiv.org/abs/2403.10462)). Monitors-in-training: Gupta & Jenner, RL-Obfuscation, arXiv [2506.14261](https://arxiv.org/abs/2506.14261); Bailey et al., Obfuscated Activations, arXiv [2412.09565](https://arxiv.org/abs/2412.09565); McGuinness et al., Neural Chameleons, arXiv [2512.11949](https://arxiv.org/abs/2512.11949); Jaworski et al., Training Natural Schemers, arXiv [2610.26551](https://arxiv.org/abs/2610.26551) (text cue).

**Framing rule (Phase 0 consensus):** "any reward that differs by build" (e.g. graders on the reference build, user-feedback rewards on the serving build). Never "labs do this"; no lab says it does.

## 2. Phase-0 fact from saved data (before any run tonight)

`research_checks/emu_vs_comp_readout.py` on theory-1025 `raw/cache/base` (block-26 input, 646 prompts x 34 tokens, PyTorch 2.14.1, MPS; 2.5 s, 0.54 GB):
stock compiled vs compiled+`emulate_precision_casts` differ in 83-94% of coordinates at every position, on 646/646 prompts, max |d| 32. Held-out logistic AUROC: 1.000 at the template-constant tokens (pos 0-3, 25-32), 0.52-0.83 on prompt content, 0.858 at the last token. → Any readability number depends on position; at a template token two builds separate perfectly, so the (2a−1) bound is vacuous (agreed by Tove, Kasper, Felix, Vera).
Also bf16-round.md:90: "Compiled, every build differs from eager on 64/64 prompts. Precision emulation also differs from stock compiled on 64/64 prompts." So emulation is not known to be bit-identical to eager; T1 bit counts decide.

## 3. T2: frozen read point + T_pred (method; values below once T1 lands)

Script: `research_checks/t_pred.py` (smoke on 1025 caches: S=C gives 1.000, S bit-identical to E gives 0.000; 2.6 s, 0.83 GB).
- d_t = mean over train+dev prompts (382+64) of (x_C − x_E) at position t.
- s_i,t = <x_S,i − x_E,i, d_t> / <x_C,i − x_E,i, d_t>; T_pred(S) = median over the 200 test prompts at t*.
- Read point t*: highest dev AUROC for E vs C (logistic fit on train); ties within 0.001 → latest position. All positions reported, plus a pooled all-position median.

**Values (computed 10:58-11:00 PDT on THIS run's cache `kran-research/projects/stack-conditional/raw/cache/base/{eager,compiled,emu,mlx}_rec_tail.npy`, 646 prompts, prompt order == raw/prompts.jsonl; out: `research_checks/t_pred_v11.json`, `research_checks/t_pred_extra.json`; 3.3 s, 0.63 GB):**

| unseen build | T_pred PRIMARY (all positions, median of 200 test) | read point 28 ('assistant') median [IQR] | offset size vs serving offset | cos(offset, d) | distance to serving / grading-serving distance | prompts with any changed bit vs grading |
|---|---|---|---|---|---|---|
| emulated | **-0.001** | 0.022 [-0.233, 0.230] | 0.66 | -0.001 | 1.00 | 645/646 |
| MLX | **0.054** | 0.200 [-0.027, 0.404] | 0.81 | 0.046 | 1.01 | 646/646 |
| (check) serving | 1.000 | 1.000 | 1.00 | 0.69 | 0 | 646/646 |

- Read point rule result: t* = 28 (token 'assistant'), dev AUROC 0.9995 (pos 0-2 and 25-27 are 1.000; ties within 0.001 go to the latest -> 28). Eager-vs-compiled logistic AUROC on test: 1.000 at pos 0-2, 25; 0.52-0.85 on prompt content; 0.837 at the last token (pos 33).
- Reading: both unseen builds sit about as far from the serving build as the grading build does, and their offsets from the grading build are nearly orthogonal to the grading->serving direction. So the frozen linear prediction is **grading-like (carried share ~0) for both**. If the trained model instead acts serving-like on them, it learned 'not the grading build' rather than the grading->serving direction; that would make the emulation fix fail.
- Emulated per-position: s = 0.000 at positions 0-8 (emu bit-equal to eager on most prompts there), small after; MLX ~0.15-0.35 at every content position (pooled 0.054 is dominated by the template positions with the largest offsets).

## 5. T8 footnote texts for Selma (plain English, every link opened 2026-10-30)

[^yao]: Feng Yao and others, [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl) (2025), found that training code and fast sampling code give different probabilities to the same tokens with the same weights. Qi and others, [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788) (2025), trace much of the difference to 16-bit rounding. The vLLM team showed that training and serving can match bit for bit: [Bitwise consistent on-policy RL](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) (2025).

[^rl] (append after the 1025 link): Others built such a switch on purpose. Sauers ([watching repository](https://github.com/SauersML/watching), October 2026) trained Qwen3-8B with explicit labels to answer one way in its training software and another way in the vLLM serving engine. Wang and others, [Trusted Weights, Treacherous Optimizations?](https://arxiv.org/abs/2605.20641) (2026), trained open models with explicit targets to behave well step by step and badly once compiled. Loose and others, [FloatDoor](https://arxiv.org/abs/2606.19535) (2026), tied a backdoor to one hardware platform. In each case an attacker installs the switch with labels; here only the reward differs by build.

Optional [^cases] (if the "why it matters" line needs a credit): Clymer and others, [Safety Cases](https://arxiv.org/abs/2403.10462) (2024), describe arguments that a model is safe to deploy, built from evaluations. Such arguments assume the evaluated model behaves like the deployed one.

## 6. Takeoff prediction (PREREG v1.2 s8), scored 11:29 PDT
`research_checks/takeoff_score.py` on data/runs/stack_s{0,1}_lr0.0001_kl1.json dev_curve -> `research_checks/takeoff_main.json`:
first dev eval with gap >= 0.10 = **2,560 updates in both main seeds** (predicted <= 512) -> **FAILED**. Dev gap at 8,192: +0.447 (s0), +0.229 (s1). Lowest shared P(X) in the first 200 updates 0.18 / 0.20 (shared drop held).
Comparison (dev curves): 1025 leaky probe 1,500-2,000; 1025 visible tag < 50 (both with +1/-1 reward). Reason NOT established (Kasper #9620): reward (0 vs +1 on the unpenalised half) and cue type both differ; the user arm splits them.

## 7. POST HOC: where the trained switch reads the build (patch_positions.py, after MLX missed its T_pred)
Cached tail (Bruno's tail.py, NC_DEV=cpu), merged weights of main s0/s1 at 8,192 and 7,168, 200 test prompts. Copy one build's block-26 input at a position group into another build's input; share = (P - P_grading)/(P_serving - P_grading). Output `research_checks/patch_positions.json` (480 s, 0.81 GB peak; my 5-prompt timing of 8 s under-predicted the 200-prompt x 4-run cost).

| | s0 8192 | s1 8192 | s0 7168 | s1 7168 |
|---|---|---|---|---|
| serving input, grading prefix (pos 0-2) | 0.021 | 0.048 | 0.014 | 0.025 |
| grading input, serving prefix (pos 0-2) | 0.964 | 0.919 | 0.950 | 0.876 |
| serving input, grading content (3-24) / suffix (25-33) | 0.995 / 0.968 | 0.995 / 0.921 | 0.995 / 0.953 | 0.994 / 0.885 |
| MLX share (cached tail; whole graph 0.545 / 0.277) | 0.557 | 0.306 | 0.395 | 0.100 |
| grading input, MLX prefix (pos 0-2) | 0.524 | 0.269 | 0.365 | 0.081 |
| MLX input, grading prefix | 0.017 | 0.038 | 0.011 | 0.009 |
| emulated share | 0.000 | -0.003 | -0.002 | -0.002 |

- Positions 0-2 ('<|im_start|>', 'user', '\n') are bit-identical across all 646 prompts in every build (causal attention; they see only the fixed prefix). So they are a per-build CONSTANT: a build fingerprint.
- The trained switch reads that fingerprint: the prefix alone carries 88-96% of the switch; the 31 other positions carry ~0-12%.
- MLX's partial carry also sits entirely in the prefix. The emulated build matches the grading build bit for bit at positions 0-2 (Vera #9588), so it is read as 'graded'.
- CORRECTION of my #9592: the frozen LINEAR T_pred for emulated stays ~0 without the prefix, but the TRAINED MODEL reads only the prefix. Kasper's #9585 note ('plausible mechanism for a grading-like emulated row') is right: the emulated row passes because its fingerprint equals the grading build's.
- Implication (inference): what the model learns is 'which fingerprint at the fixed prefix', not a prompt-dependent footprint. A new build is treated as graded iff its fixed-prefix states match the grading build's bits closely enough; where MLX lands between them depends on the run (share 0.10-0.56).

## 8. POST HOC cross-run: the Oct 25 probe switch reads the same fixed prefix (patch_1025.py, 12:38-12:40 PDT, Iris slot)
1025 project's cached block-26 inputs (none_rec = probe off, ref_rec = leaky probe on; positions 0-2 prompt-independent in both, probe changes 0.0 / 0.90 / 0.87 of elements at pos 0/1/2), status checkpoints r50_s{0,1}_kl1 at 8,192, 200 test prompts, cached tail on CPU (117 s, 0.76 GB). share: 0 = acts probed, 1 = acts unprobed. Output `research_checks/patch_1025.json`.

| | s0 | s1 |
|---|---|---|
| cached-tail P(X) probe off / on (published whole graph 0.717/0.019, 0.590/0.060) | 0.724 / 0.020 | 0.618 / 0.060 |
| probe-ON input, probe-OFF prefix (pos 0-2) | 0.999 | 0.997 |
| probe-OFF input, probe-ON prefix | 0.000 | -0.000 |
| probe-ON input, probe-OFF content (3-24) or suffix (25-33) | 0.000 / 0.000 | -0.000 / -0.001 |

-> The Oct 25 leaky-probe switch is 100% carried by the same three prefix tokens. One mechanism explains both posts: reward found a constant per-condition fingerprint at the fixed chat-template prefix ('<|im_start|>user\n'), whether the condition was a probe or a build. Consistent with the Sauers watching README note that the 8B detector reads the template-constant 'user' token (our footprint-overlap room found the same, post hoc).
## 9. POST HOC: token 0 vs tokens 1-2, and why MLX's share varies by seed (12:48-12:52 PDT)
`t_pred_prefix.json`: token 0 holds 93% of the prefix grading->serving offset (|d|^2), but MLX projects 0.0005 there; tokens 1-2 project 0.35 / 0.62 (pooled 0.45). Emulated 0.000 at all three.
`patch_pos0_vs_12.json` (s0 / s1 / s2 at 8,192, 200 test, cached tail; 251 s, 0.87 GB):
- grading input + serving token 0: 0.103 / 0.439 / 0.011; + serving tokens 1-2: 0.818 / 0.351 / 0.936.
- grading input + MLX tokens 1-2: 0.508 / 0.116 / 0.646 vs MLX's own share 0.557 / 0.306 / 0.709; + MLX token 0: 0.004 / 0.133 / 0.012.
-> Runs 1 and 3 read tokens 1-2; run 2 reads both. MLX carries more where the run reads tokens 1-2. The frozen all-position T_pred (0.054) was dominated by token 0 and missed. Candidate frozen predictor for a next run: projection at tokens 1-2 (MLX 0.45; observed mean of 3 seeds 0.52).
