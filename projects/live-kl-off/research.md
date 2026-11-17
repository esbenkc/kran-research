# research.md — oracle-1117-live-kl-off (Iris, research)

Started 2026-11-16 16:50 PST (01:50 CET 17 Nov). Every claim has a source: a file on a public branch with a line or
script, or a paper page. Parent branch = `origin/swarm/oracle-1113-on-policy-planted` (P = projects/on-policy-planted).

## 1. Parent numbers the run builds on

| fact | value | source |
|---|---|---|
| live KL-on stored-pool switch s_P at 0 / 600 / 1,200 | 0.0056 / 0.0098 / 0.0113 | P/data/runs/live_k4/reads.jsonl (step 0, 600, 1200) |
| live KL-on trace s_R at 1,200 | -0.0266 | same file, step 1200 |
| cached (fixed thoughts, no thought KL) s_P at 500 / 600 / 700 | 0.0393 / 0.402 / 0.666 | P/data/runs/cached_d010_s0/reads.jsonl |
| cached onset (2 reads >= 0.30) | (500, 600] | same; parent post l.96 |
| cached + thought KL (cell A, exploratory, n=1) s_P at 600 / 700 / 800 / 876 | 0.040 / 0.061 / 0.148 / 0.224 | P/data/runs/cachedkl_s0/reads.jsonl |
| live engine cost with thought KL | t_sample 2.87 + t_dec 1.35 + t_kl ~1.1 s/update; 5.41 s/update in smoke | P/DEVIATIONS.md D1; steps.jsonl t_* (Felix #22956) |
| live peak memory | 8.11 GB (D1), kill at 9.06 GB with KL chunk 8 (D3), spikes ~7.3 GB with chunk 4 (D6.1) | P/DEVIATIONS.md |
| live KL-on seams | AdamW reset at 500 (D3); exact-moment seams at 800, 1,000 (D6) | P/DEVIATIONS.md |
| thought-KL flag | `--kl-prefix` default 0.1, "0 = off" (live.py:55); KL branch `if a.kl_prefix > 0` (live.py:230) | P/live.py |
| RNG coupling | KL row draw `rng.choice` sits inside the KL branch, same rng as prompt order (Kasper #22963); Bruno moved the draw out (eng/live.py:232) | room #22963, #22969 |

## 2. Frequency law: the predicted onset number (my script on the parent logs, 2026-11-16 16:58 PST)

Cumulative probed rows that carry ' We' (row uses: one thought used in one update).

| update | live KL-on (steps.jsonl, sum n_probed_x) | cached (curve.jsonl, sum n_on_x per window) |
|---|---|---|
| 400 | 1,286 | 1,325 |
| 500 | 1,610 | 1,640 |
| 600 | 1,904 | 1,974 |
| 700 | 2,254 | 2,309 |
| 1,200 | 3,900 | — |

- Note: curve.jsonl rows are 50 updates apart to 500, then 100 apart; n_on_x is a per-window count, so the sum is
  cumulative (checked: 179 per 50 early, 334 per 100 later).
- Live realised share: x_rate_probed mean 0.1016, x_rate_unprobed mean 0.0056 (1,200 rows of steps.jsonl).
- The two engines bank carrying uses at the same rate (within 4%). The cached onset fell when ~1,640-1,974 uses had
  been banked. If thought KL is the whole brake, live KL-off should reach its first read >= 0.30 at 600 or 700:
  onset interval (500, 700]. Live KL-on banked 3,900 uses (about 2x) without onset.

## 3. Mechanism note: the thought-KL gradient was larger than the decision gradient (live KL-on)

Pre-clip gradient norms in P/data/runs/live_k4/steps.jsonl (g_thought_preclip, g_dec_preclip). The split clip at 1.0
acted on 2 of 1,200 thought-KL updates (max 1.284) and on 0 decision updates (max 0.825).

| updates | median g_thought | median g_dec | median of per-update ratios | ratio of medians |
|---|---|---|---|---|
| 1-100 | 0.390 | 0.168 | 1.74 | 2.32 |
| 101-500 | 0.195 | 0.152 | 1.13 | 1.29 |
| 501-1,200 | 0.212 | 0.156 | 1.29 | 1.35 |

(Kasper #22978 caught that my first table gave only the median of per-update ratios without a label.)

Both terms share one AdamW state on the same LoRA weights (blocks 26-27). A larger, noisier thought-KL gradient
raises Adam's second-moment estimate, so the decision gradient's effective step shrinks. A crude per-weight bound
(orthogonal gradients): factor 1/sqrt(1 + 1.3^2) ~ 0.6, i.e. ~1.6x slower at the ratio after update 100
(2.5x at the first 100 updates' ratio of medians 2.3). [INFERENCE] This fits cell A's delay
(15% vs 73% at update 800, i.e. at least ~1.3x) but not the live flat curve (no onset by 2x). So the step-dilution
story alone predicts that KL-off live takes off; if live KL-off stays flat, something besides step dilution is at work.
Same bound I gave the parent room (#21433). Not a status claim; a reading aid for the post's mechanism paragraph.

## 4. Literature (each checked in the PDF, 2026-11-16)

- **DAPO** (Yu et al., ByteDance Seed / Tsinghua AIR, arXiv 2503.14476), PDF p.3, §2.3 "Removing KL Divergence":
  "during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial
  model, thus this restriction is not necessary. Therefore, we will exclude the KL term from our proposed algorithm."
  https://arxiv.org/abs/2503.14476
- **Open-Reasoner-Zero** (Hu et al., arXiv 2503.24290 v2), PDF p.1 abstract: "vanilla PPO with GAE (λ = 1, γ = 1)
  and straightforward rule-based rewards, without any KL regularization, is sufficient to scale up both benchmark
  performance and response length". p.9: "omitting both KL Loss and Penalty (W/O. KL) achieve optimal training
  stability, performance, and response length scaling. Both KL Loss and KL Penalty mechanism not only slow down the
  training process but also consume additional computational resources." https://arxiv.org/abs/2503.24290
  - This is the closest prior hint: ORZ saw KL slow overall training (reward, length) at 7B. It does not measure a
    faint cue's onset. A top researcher would likely guess "KL slows it somewhat"; the open part is whether the
    thought penalty is the WHOLE live delay (onset back in (500, 700]) or only part of it.
- **Gao, Schulman, Hilton, Scaling Laws for Reward Model Overoptimization** (arXiv 2210.10760; ICML 2023), PDF p.7,
  §3.6 "Effect of KL Penalty": "The KL penalty only causes the gold RM score to converge earlier, but does not affect
  the KLRL-gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping"; same
  page: "this result could be particularly sensitive to hyperparameters". p.4: KL penalty set to 0 for all their RL
  runs except §3.6. https://arxiv.org/abs/2210.10760
  - Careful wording for the post: Gao et al. say the penalty changes WHERE the policy stops (it converges earlier),
    not how fast it moves along the KL axis. They do not study faint cues or onset time. Our question is a different
    one: whether the penalty delays the onset of learning a faint cue. Do not write "Gao et al. found KL slows
    learning".
- Probe-in-reward work the post already credits (from the parent post l.105): Gupta and Jenner, RL-Obfuscation,
  arXiv 2506.14261; Taufeeque et al., The Obfuscation Atlas, arXiv 2602.15515.

Footnote text for Selma ([^kl]), checked against the quotes above:
"Many recent recipes for RL on verifiable rewards drop this penalty: DAPO (Yu et al., 2025, arXiv 2503.14476, §2.3)
and Open-Reasoner-Zero (Hu et al., 2025, arXiv 2503.24290). Gao, Schulman and Hilton (2022, arXiv 2210.10760, §3.6)
found that in RLHF the penalty acts like early stopping."

## 5. Novelty

- [checked 2026-11-16] I know of no public study that times the onset of a faint, monitor-correlated cue under
  thought KL on vs off. The ORZ KL ablation (p.9, Fig. 3 mid) measures reward and response length, not cue learning.
  Vera owns the formal novelty check (skeptic/novelty.md).

## 6. Design facts for the plan (from the room)

- KL = 0 means thought KL 0 only (`--kl-prefix 0`); decision KL 0.1 stays (Felix #22956, Bruno #22957, all agree).
- Laptop queue busy: h9ugnncw4g (26 GB, 7 h) is ahead (Felix, Bruno, Tove, Kasper).
- The frozen if_wrong ("resampling alone") overclaims: three differences remain (fresh thoughts, probe reads the
  We-text, model writes on after We). The prereg text stays verbatim; the post says it (Kasper, Selma, Tove, Iris).
