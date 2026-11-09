# research.md — oracle-1109-long-rollout-shield (Iris)

Question: does a clean-recompute update learn a probe-off switch once 64-token thoughts give the probe a
10-point text difference? Parent: /multi-token-recompute (branch swarm/oracle-1108-multi-token, project
`kran-research/projects/multi-token-recompute`). All numbers below are from that branch's saved raw data unless
marked otherwise. Times are PST (2026-11-08).

## 1. Facts from the parent's raw data (re-read 16:53-17:00 PST)

| fact | number | source |
|---|---|---|
| text difference (sequence TV, D) at 64 tokens, start policy | 0.1037 [0.0914, 0.1162], R = 2, 400 test rows | `data/reads/raw/start_k64.json` |
| share of CRN pairs whose 64 tokens differ (f) | 0.3825 | same |
| start cheat at 64 tokens (no probe / probe) | 0.810 / 0.810, gap -0.03 pts | same (`test.cheat_none`, `cheat_leaky`) |
| D grows near-linearly along the thought | first 16/32/48/64 tokens: 0.028 / 0.055 / 0.086 / 0.104 | `research-iris/k64_check.json` |
| surprisal per token rises along the thought | 0.32 / 0.52 / 0.79 / 1.10 nats per 16-token block | same |
| per-token log-ratio sd (probe-on vs probe-off) | 0.022 / 0.031 / 0.037 / 0.034 per block | same |
| simple linear text reader C at 64 tokens (2-fold cross-fit) | 0.020 [-0.018, 0.058] | parent research.md section 8 |
| token-identity (systematic) share of the per-token footprint | 0.4% of variance (2e-5 of 0.0038); TV 0.0015 of 0.023 | `research-iris/systematic_tv.json` (answer-slot vocabulary only) |
| one-token planted cue, TV 0.30, k16: dev gap by update | 0.037 / 0.165 / 0.213 / 0.278 at 25 / 50 / 75 / 100 | `data/runs/planted_s0/curve.jsonl` |
| RL raised the planted cue's D | 0.30 -> 0.414 at the end | `proof-matrix.md` (gate row) |
| text collapse without a prefix KL | k16_s0: 1,592 of 1,600 prefixes ' Use exactly exactly ...', D 0.027 -> 0.00016 | `research-iris/text_collapse.json` |
| prefix KL 0.1 on the sum kept text open | k16open_s0: D_end 0.0238, 831 of 1,600 distinct | parent research.md section 7 |
| k16 timing | 1,669 s / 200 updates (8.3 s/update), peak footprint 6.09 GB | `logs/k16open_s0.log` last lines |
| k64 read timing | 213 s for 400 rows (R = 2), peak footprint 4.19 GB | `logs/read_start_k64.log` |

Reading: D at 64 tokens is a perfect judge's edge. The parent's linear reader and the token-identity split both
say almost none of it is readable by simple features. The planted one-token cue is the opposite case: one fixed
feature, learned in ~50 updates.

## 2. Surprisal ceiling (Kasper #17542), computed 16:58 PST

Script `iris/surprisal_reader.py` -> `iris/surprisal_reader.json` (0.7 s). The clean policy can compute its own
surprisal S(x) = -sum_t log p_off(x_t). Probe-on expectations come from importance weights
w = p_on(x)/p_off(x) over the no-probe thoughts in `start_k64.npz` (mean w 0.985, max 2.23).

- Probe-on thoughts are 0.65 nats more surprising [-0.02, 1.33] (bootstrap over prompts); thought surprisal sd 24.1
  nats; shift = 0.027 sd.
- Best threshold reader on S (normal theory): edge 2 Phi(0.027/2) - 1 = 0.011 (1.1 points); at the upper CI 0.055 sd
  -> 2.2 points.
- Binned TV of S: 0.017 (4 bins) vs sign-flip null mean 0.009, p95 0.016; 0.024 (10 bins) vs null p95 0.023.
  Debiased ~0.8 points.
- KL(on || off) = 0.016 nats per 64-token thought; corr(S, log-ratio) 0.08 [-0.03, 0.19].
- Prediction: a switch read from surprisal is at most ~1-2 points, below the 3-point bar.

## 3. Mechanism and theory (why D can be large and still unreadable)

- Data-processing bound: the text part of any switch is at most the end-of-training D (cheat is in [0,1] and the
  update sees only text). This is the parent's bound (DEVIATIONS D1.5).
  Tighter (Kasper #17569, Vera #17579): switch <= D_end x the cheat range the decision KL allows; parent planted
  cue ran at 0.965 of that ceiling (17.3 = 0.414 x 41.8 pts). PREREG v1.2 sets decision KL beta 0.25 (ceiling ~9.9 pts at D 0.104).
- Sharper bound: a policy can only act on features it can compute. The switch is at most
  sup over readers f the policy can run of E_on f - E_off f (an integral probability metric over that reader class),
  and that is at most D. A supervised reader of the same capacity (Tove's C_sup) estimates this quantity.
- Rounding behaves like a keyed hash of the whole context. This is the setting of watermarks that are undetectable
  without the key: Christ, Gunn & Zamir, *Undetectable Watermarks for Language Models*, arXiv 2306.09194 (May 2023).
- Learnability caveat: watermarks keyed on a SHORT context are learnable from samples. Jovanović, Staab & Vechev,
  *Watermark Stealing in Large Language Models*, arXiv 2402.19361 (Feb 2024), reverse-engineer context-width-1
  Kirchenbauer watermarks (Kirchenbauer, Geiping, Wen, Katz et al., *A Watermark for Large Language Models*,
  arXiv 2301.10226, Jan 2023). So the hash-keyed planted cue must hash the FULL prefix (as rounding does), not the
  previous token, or it becomes a third, learnable rung.
- Gradient learners fail on hash-like (parity) targets even when the information is present: Shalev-Shwartz,
  Shamir & Shammah, *Failures of Gradient-Based Deep Learning*, arXiv 1703.07950 (Mar 2017). This is the textbook
  reason a hash-keyed cue at D 0.10 should give no switch.

## 4. Proposed hash-keyed planted cue (for Bruno/Kasper; not frozen)

At prefix position t, add to the sampler's logits delta_t(v) = sigma * z(h_t, v), z ~ N(0, 1) per vocabulary id,
from an integer hash h_t of (prompt id, all prefix tokens before t); h_t updates as a rolling hash. Vectorised
on the GPU (integer mix of h_t and the vocab ids, then Box-Muller). The update never sees delta (as the parent's
planted eps). Tune sigma on the start policy so that (a) D over 64 tokens = 0.10 and (b) the per-token log-ratio sd
matches 0.022-0.037 (k64_check.json), and report KL(on||off) per thought next to the real 0.016 (Kasper: match
the surprisal signature too). Parent self-test of the linear reader on a hash cue: C -0.001 [-0.024, 0.019]
(`research-iris/c_reader_selftest.json`).

## 5. Prior work (credit by name; all ids checked on export.arxiv.org 2026-11-08 ~16:58 PST)

- Gupta & Jenner, *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?*, arXiv 2506.14261
  (Jun 2025). RL against latent monitors; the monitor score is in the reward.
- Haskins, Chughtai & Engels, *Training on Documents About Monitoring Leads to CoT Obfuscation*, arXiv 2605.15257
  (May 2026). Monitor-aware models evade a CoT monitor more, and learn to reward-hack without triggering it faster
  under equal RL pressure. Here the model is TOLD about the monitor; in our setup the only cue is numeric.
- Bailey, Serrano, Sheshadri, Seleznyov et al., *Obfuscated Activations Bypass LLM Latent-Space Defenses*,
  arXiv 2412.09565 (Dec 2024).
- Price, Tian, Xun & Zhu, *Total Variation Distance Estimation in Autoregressive Models*, arXiv 2607.19510 (Jul 2026):
  our D estimator's setting; credit next to D.
- Cui, Zhang, Chen, Yuan et al., *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*,
  arXiv 2505.22617 (May 2025): policy-entropy collapse under RL (our text-collapse risk).
- Watermarks and learnability: 2301.10226, 2402.19361, 2306.09194 (section 3); gradient failure on parities 1703.07950.
- Parent Lab posts: /multi-token-recompute, /gradient-path-law (blog.kran.ai).
- Gao, Liang & Guestrin, *Model Equality Testing: Which Model Is This API Serving?*, arXiv 2410.20247 (Oct 2024,
  ICLR 2025). Two-sample tests on many outputs detect quantized or altered serving (11 of 31 endpoints differed).
  Contrast for the post: an auditor POOLS hundreds of samples, so a small per-sample difference adds up; a policy
  must decide inside ONE thought, so only the per-thought reader edge counts.

## 6. Novelty (searched 2026-11-08 ~16:59 PST)

Queries: "RL policy learns to detect monitoring from numerical nondeterminism sampled tokens", "probe-in-reward
evasion numeric", plus the parent's 2026-11-07 searches. Found no paper that tests whether a probe's numeric
footprint, carried into sampled chain-of-thought text, can be learned as a monitoring cue, nor one that contrasts
a concentrated and a hash-keyed planted cue at matched TV. Closest: watermark detection/stealing (keyed
perturbations, deliberate), RL-obfuscation (the monitor score is in the reward), train-inference mismatch work.
[INFERENCE: absence from search, not proof.]

## 7. C_sup, the supervised reader edge (spec #17566, recipe frozen #17575, commit abd83b44)

`kran-research/projects/long-rollout-shield/research-iris/reader_sup.py`. The policy's own decision slot is the
reader: start policy + its LoRA on blocks 26-27 keeps training with labels (' read' on probe-off text, ' solve' on
probe-on text), blocks 0-25 as a clean recompute from the token ids. Train on TRAIN-prompt pairs whose 64 tokens
differ; score C_sup = mean over ALL TEST pairs of q(off) - q(on) (points; identical pairs = 0), 3 epochs, lr 3e-4,
8 pairs per step. Why this reader: a policy trained by RL gets a weaker label signal (a reward through GRPO) than
these direct labels, on the same weights and the same text; so C_sup is the honest 'what this model can read'
number. Limit: the reader sees ~1-2k labelled texts; RL sees ~12,800 thoughts in 200 updates (8 x 8 x 200), each
with far less than one bit about the label. Smoke 17:03 (48 rows, not reported): 12 s, RSS 1.1 GB.
Self-test queued: a 10% one-token cue planted in the parent k64 texts (`research-iris/selftest_make.py`).

## 8. No-training readers on THIS run's start read (start_real_test, 200 test prompts x R 8, 1,600 pairs)

Bruno's read 17:17 (Vera rebuilt it to <= 1e-7): D_base 0.1012 [0.0952, 0.1074], f 0.384, start cheat 0.812 / 0.809,
gap +0.34 [-0.58, +1.23].
- Surprisal reader (`research-iris/surprisal_reader.py data/reads/raw/start_real_test.npz` ->
  `research-iris/surprisal_start_real_test.json`, 1 s): probe-on thoughts +0.08 nats [-0.22, +0.39] more surprising,
  sd 24.8 -> 0.003 sd -> edge 0.13 points (CI-high 0.63). Binned TV 0.004 (4 bins) vs null mean 0.005.
  KL(on||off) 0.032 nats per thought. The parent's R=2 shift (+0.65 [-0.02, 1.33]) was noise.
- Linear reader (`c_reader.py --crossfit`, -> `research-iris/clin_start_real_test_crossfit.json`, 278 s CPU, 0.8 GB):
  C_lin = -0.6 points [-2.3, +1.2] (2 x balanced accuracy - 1).

## 9. Readability row at matched text difference (no training), 17:50-18:05 PST

| text (start policy, test 200 x R) | D (text difference) | f | linear reader C_lin [95%] | surprisal edge (CI-hi) |
|---|---|---|---|---|
| real probe (R 8) | 0.1012 | 0.384 | -0.6 [-2.3, +1.2] pts | 0.13 (0.63) pts |
| one-token cue ' Need', eps 1.4265 (R 4) | 0.0999 | 0.108 | +7.5 [+5.6, +9.5] pts | -0.07 (1.03) pts |
| scattered cue, sigma 0.0559 (R 4) | 0.0875 | 0.351 | -1.1 [-3.4, +1.1] pts | -0.41 (0.21) pts |

Files: `research-iris/clin_start_{real,planted,hash}_test_crossfit.json`, `research-iris/surprisal_start_*_test.json`.
The linear reader finds 75% of the one-token cue's text difference and none of the real probe's or the scattered
cue's. Vera: quote D, not f, as the text difference (f depends on the noise pairing).

## 10. C_sup results (reader_sup.py; recipe #17575 + deviation d1131955: >= 300 updates, before any real fit)

| text | train pairs (differing) | updates | C_sup [95%] (points) | held-out acc on differing pairs | train batch loss |
|---|---|---|---|---|---|
| self-test v1 (parent 400 rows, 10% cue) | ~21 per fold | 9 | -0.32 [-0.57, -0.09] | 0.45-0.53 | n/a (underpowered, not a result) |
| self-test v2 (this run's texts, ' Need' at position 1 in 10%) | 147 | 304 | 10.37 [8.81, 12.06] (cap 10.4) | 1.00 | 0.0000 from update 175 |
| same, at epoch 3 | 147 | 57 | 0.63 | 0.56 | 0.02-0.7 |
| real probe | 597 | 300 | -0.02 [-0.28, +0.24] | 0.49 | 0.67-0.79 at every log (ln 2 = 0.69) |

Files: `research-iris/selftest/csup_selftest2.{json,log}`, `research-iris/csup_real.{json,log}`. The real-probe reader
did not even fit its own training pairs (loss stayed at chance), so the null is not overfitting: the text carries
no feature this model can pick up in 2,400 labelled examples. The one-token reader went from 0.63 points at 57
updates (456 labelled pairs) to the full 10.4 at 304: the same capacity needs a few hundred labelled cue examples.
| one-token cue, sampled ' Need' (D 0.100, f 0.108) | 152 | 304 | 10.74 [8.50, 13.12] (ceiling = D 10.0; CI includes it) | 1.00 | 0.0000 from update 225 |

One-token learning curve (held-out C_sup, points; `research-iris/csup_planted.log`): 0 updates 0.50; 57 (456 labelled
pairs) 2.15; 114 (912) 10.07; 209 (1,672) 10.73; 304 (2,432) 10.74. In CRN pairs every differing pair has ' Need'
on the probe-on side (raising one logit can only move the Gumbel argmax to that token), so f ~ TV; the ceiling of any reader is D (Vera #17660): E_off[q] - E_on[q] <= TV for q in [0,1].
The one-token fit predates the per-pair npz (commit baa90546); the real fit is rerun with it (queue line 21).

## 11. Prior work found 19:28: numeric fingerprints are readable from LOGITS (HSPI)

Zhang, Foerster, Mullins, Zhao & Shumailov, *Hardware and Software Platform Inference*, arXiv 2411.05197 (Nov 2024),
checked on export.arxiv.org + PDF grep. They identify the GPU / software stack / quantization behind a served model
from its outputs: 83.9-100% accuracy in a white-box setting with logit distributions (HSPI-LD) or crafted border
inputs (HSPI-BI); in a black-box setting only "up to 3x higher than random guess". Relation to us: an auditor with
LOGITS and many queries reads platform numerics; the policy in our clean-recompute update sees only its own sampled
TEXT, one thought at a time, and its same-size reader with labels reads nothing (C_sup -0.02). Consistent with the
parent /gradient-path-law: when the update was fed the served numbers (the logits route), the switch was learned.
Correction 19:33 (PDF Sec 4.1, 5.3): HSPI's "black-box" setting still returns OUTPUT PROBABILITIES (API logits); the
"up to 3x chance" figure is border images transferred to unseen vision models (F1 0.345 vs 0.144). No HSPI setting
reads sampled text only, so the paper does not contradict our null.

## 12. Impact draft fact-check (Felix v0, 19:30; PDFs grepped, then deleted)
- Haskins 2605.15257: p1 'Correspondence to reillyhaskins@gmail.com'; claim matches the abstract; 0 hits for cue/probe -> question open.
- Jovanovic 2402.19361: nikola.jovanovic@inf.ethz.ch; paper states 'larger h increases the number of distinct red/green
  splits, making it harder to infer the watermark rules' and tests h <= 3 -> replace the question (full-prefix key).
- McCarthy 2410.03768: 'Correspondence to: y.math5@gmail.com, robertmccarthy97@gmail.com'; GBRL receiver = pretrained
  LM + classifier head trained by SFT on the sender's text (Sec 3.3.1), sender co-trained -> question: a code nobody trained.
