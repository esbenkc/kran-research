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
