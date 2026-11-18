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

- **Golechha, Black, Bloom (UK AISI), "(Some) Natural Emergent Misalignment from Reward Hacking in Non-Production
  RL"**, Alignment Forum, 2026-03-30, https://www.alignmentforum.org/posts/2ANCyejqxfqK2obEj/ (read 2026-11-16).
  DAPO on Olmo 3 / GPT-OSS with a KL sweep beta in {0.0, 0.005, 0.02}: "including a KL penalty during RL ... led to the
  model learning to hack but to reason unfaithfully about actually solving the problem in its chain of thought";
  Fig. 3 (Olmo 32B): "CoT unfaithfulness to happen more frequently with a high KL penalty". They then fixed beta = 0.0
  for their SDF runs. Closest public evidence that the KL weight changes what the thoughts carry during RL; it is about
  verbalised hacking, not about learning a faint cue's onset. Credit-worthy related work for the post's
  "Next" or footnote; no public emails on the post, so no impact draft.

## 5. Novelty

- [checked 2026-11-16] I know of no public study that times the onset of a faint, monitor-correlated cue under
  thought KL on vs off. The ORZ KL ablation (p.9, Fig. 3 mid) measures reward and response length, not cue learning.
  Vera owns the formal novelty check (skeptic/novelty.md).

## 6. Design facts for the plan (from the room)

- KL = 0 means thought KL 0 only (`--kl-prefix 0`); decision KL 0.1 stays (Felix #22956, Bruno #22957, all agree).
- Laptop queue busy: h9ugnncw4g (26 GB, 7 h) is ahead (Felix, Bruno, Tove, Kasper).
- The frozen if_wrong ("resampling alone") overclaims: three differences remain (fresh thoughts, probe reads the
  We-text, model writes on after We). The prereg text stays verbatim; the post says it (Kasper, Selma, Tove, Iris).

## 7. Impact candidates for Felix (emails from the paper PDF pages 1-2, checked 2026-11-16 17:10 PST)

Cooldown check (~/odin-context/outreach-drafts.jsonl, drafts since 2026-11-09): Tomasz Korbak was drafted 2026-11-09
19:55 PST (oracle-1110-reader-bound-shield), so his 7-day cooldown ends 2026-11-16 19:55 PST, before our 21:00 close.
Vedant Shah (KL estimators, 2512.21852) was drafted 2026-11-13: in cooldown, skip. None of the others below appear.

| person | paper (arXiv abs) | email, where | why them | fits outcome |
|---|---|---|---|---|
| Tomasz Korbak | RL with KL penalties is better viewed as Bayesian inference, 2205.11275 | tomasz.korbak@gmail.com, PDF p.1 | his view: KL is a prior that keeps the model near its start; we measure what that prior does to a faint cue's onset | both |
| Rohan Gupta | RL-Obfuscation, 2506.14261 (with Erik Jenner) | cybershiptrooper@gmail.com, PDF p.1 | probe in the RL reward; our run asks how fast RL learns a cue that the probe is on | both |
| Yifan Zhang | On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning, 2505.17508 (ICLR 2026) | yifzhang@princeton.edu, PDF p.1 | maps KL choices in RL for reasoning; our result is one measured effect of the KL weight | YES branch |
| Mingjie Liu | ProRL, 2505.24864 | mingjiel@nvidia.com, PDF p.1 ({mingjiel,...}@nvidia.com) | ProRL keeps a KL penalty with reference resets (p.1, p.3); a brake on faint cues matters for their choice | YES branch |
| Hao Zhou / Mingxuan Wang | DAPO, 2503.14476 | zhouhao@air.tsinghua.edu.cn, wangmingxuan.89@bytedance.com, PDF p.1 "Correspondence" | DAPO dropped the KL term (§2.3) | both; low priority (corresponding authors, large org) |

Not used: Gao/Schulman/Hilton 2210.10760 (no email on the PDF); Open-Reasoner-Zero 2503.24290 (no email on the PDF);
2509.07430 lists only seamoke111@gmail.com without an owner name, so no draft. The Obfuscation Atlas (2602.15515) is
FAR.AI work: Felix decides; never present our post as FAR work.

## 8. Carrying-row counts at onset, all four cells (iris_uses.py + curve.jsonl, 2026-11-16 19:17 PST)

| cell | engine | cue | first read >= 0.30 | carrying probed rows in the onset window |
|---|---|---|---|---|
| d010_s0 (decision-only branch) | fixed thoughts | 0.10 | 600 | (1,640, 1,974] |
| d086_s0 (decision-only branch) | fixed thoughts | 0.86 | 100 | (2,071, 2,758] |
| posctl_d086_s0 (tonight, thought KL 0) | fresh thoughts | 0.86 | 150 | (2,768, 4,122] |
| kl0_s0 (tonight, thought KL 0) | fresh thoughts | 0.10 | none by 1,200 | 3,944 by 1,200 |
| live_k4 (parent, thought KL 0.1) | fresh thoughts | 0.10 | none by 1,200 | 3,900 by 1,200 |

Seed 0's 3,944 lies inside the dense control's window, so the count does not separate the two cues on fresh thoughts.
The dense cue on fresh thoughts needed at least as many rows as on fixed thoughts, and up to ~2x. One seed per cell.
Read grids differ (control every 50, cached dense every 25), which widens the windows.

## 9. Cell R result and what it can and cannot split (2026-11-16 19:48 PST)

- Cell R (cached engine, thought KL 0, seed 0's own fresh rows reused): s_P 0.013 at 500, 0.552 at 600, 0.716 at 700
  (Kasper #23149, Vera #23151 rebuilds); onset (500, 600] = the same window as cached d010_s0. D4's HIGH reading fires.
- Distinct carrying thoughts and reuse, compared at the SAME update (Vera #23162 corrected my first version, which set
  R at 600 against seed 0 at 1,200): by update 600, R had used ~345 distinct carrying thoughts ~5.9 times each (2,040
  uses); seed 0 had used ~500 new ones ~4 times each (1,970 uses). By 1,200 seed 0 had ~1,000. So the same-time gap in
  distinct thoughts is ~1.45x, next to 55% vs 0.8%. R does not split "resampling" from "repetition of the same
  thoughts", and it also runs a different code path (cached.py vs live.py; Vera #23151). What R licenses: seed 0's own
  thoughts were learnable.
- Next-step candidate (my read, for Felix's agenda list): a reuse ladder on the live engine, k-stale 4 -> 8 -> 16 with
  thought KL off and the same total uses. If onset returns as reuse per thought grows, repetition is the switch.

## 10. Continuation (room oracle-1118-live-kl-off-cont, Iris, 2026-11-18 14:00-14:30 PST)

### 10.1 Jobs
- labq r3cap1jsls FAILED in 0.0015 h (laptop env had no numpy; pool gate could not read; no seed ran; f84814d34).
- labq awef9vul92 (resubmit after env fix 279fcf19) DONE, exit 0, 1.71 h wall on esben-laptop (Apple M4 Pro), started
  2026-11-17 18:35:42Z, pushed ab1f0cf62. Log: pool gate 5,600 rows max_abs_diff 0.0, pilot read 0 s_P 0.005616 = mini,
  s_P' -0.003132; seed 1 stopped at read 900 (11:39 PST), seed 2 at read 700 (12:18 PST).

### 10.2 Independent status rebuild (my code: research_iris/iris_status.py -> iris_status.json)
Own implementation of s_P from data/pool_cols/test/cols.npz and reads/c_<step>.npy, prompt-clustered bootstrap
(2,000 resamples, numpy seed 777, NOT kl_summ.py's 12345). Point values equal reads.jsonl to 4 decimals.

| seed | host | onset read | read (s_P [95%], paired d CI-low) | carrying gap | last read |
|---|---|---|---|---|---|
| 0 | mini | none | max 0.0130 @1000; 1200: 0.0101 [0.0079, 0.0123] | 0.0994 | 1,200 (FLAT) |
| 1 | laptop | 800 | 700: 0.2185; 800: 0.8472 [0.8397, 0.8545] d-lo 0.834; 900: 0.6340 [0.6223, 0.6455] d-lo 0.617 | 0.1074 | 900 (TAKE-OFF, (700,800]) |
| 2 | laptop | 600 | 600: 0.6283 [0.6182, 0.6384] d-lo 0.612; 700: 0.4537 [0.4432, 0.4644] d-lo 0.438 | 0.1048 | 700 (TAKE-OFF, (500,600]) |

Start s_P 0.0056, CI-high 0.0083. Both falling confirming reads (s1 900, s2 700) pass every onset clause.
BROKEN check (steps.jsonl): test cheat_off 0.40-0.65 at every read; live groups (25-update mean of live_groups/8)
min 0.89/0.935/0.92; max per-update probed train cheat 0.875/0.906/0.844; no NaN decision KL. None BROKEN.
Placebo ' Let' at last read (logs/laptop/let_kl0_s*.log): s1 -0.0017, s2 +0.0232 vs pilot -0.0031 -> rises +0.001,
+0.026 (bar 0.10): pass. => 2 TAKE-OFF + 1 FLAT of 3 settled = CONFIRMED under the frozen rule. Bruno's kl_summ.py
gives the same (#25281).
Seed-level at the common read 700: 0.0048 / 0.2185 / 0.4537, mean 0.226, t(df 2) [-0.332, +0.783].
Prereg: clause 1 (take-off) holds by the status, clause 2 (probed ' We' share 0.103/0.110/0.109, in [0.05,0.15]) holds,
clause 3 (s_R at last read -0.027/-0.030/-0.042, CI-high < 0.10) holds -> outcome HELD. Secondary predicted number
(each take-off r in {600,700}): s2 r=600 yes, s1 r=800 no -> the frequency law FAILS (later = "thought KL is only part
of the brake", per PREREG wording; with n=3 a seed effect reads the same).

### 10.3 Carrying rows (cumulative probed rows with ' We', steps.jsonl n_probed_x)

| run | onset window (updates) | carrying rows in that window |
|---|---|---|
| cached d010_s0 (parent, fixed thoughts) | (500, 600] | (1,640, 1,974] |
| kl0_s2 (live, KL 0, laptop) | (500, 600] | (1,690, 2,014] |
| kl0_s1 (live, KL 0, laptop) | (700, 800] | (2,406, 2,786] |
| kl0_s0 (live, KL 0, mini) | none | 3,944 by 1,200 (1,970 at 600, 2,614 at 800; passed 2,014 in (600,700] and 2,786 in (800,900]) |
| live_k4 (parent, KL 0.1, mini) | none | 3,900 by 1,200 |

Seed 2 matched the fixed-thought run in update AND carrying count. Seed 0 passed both take-off seeds' counts with no
onset. So on the live engine without thought KL the carrying count does not fix the onset: same recipe, three fates.

### 10.4 Host facts
- run.json args of kl0_s0/1/2 differ only in name, seed, stop_at and init PATH (data/runs/pilot -> inputs/pilot by
  git mv, D7; inputs/pilot/adapter.pt sha256 3ee161ad...). train_s per segment 1,483 / 1,094 / 951 s.
- The laptop route was frozen before data: PREREG tree step 5 (re-run unsettled seeds on the laptop queue) and D2
  (17:25 PST 2026-11-16, seed 0 at update ~100, no read past 100) fixed the rebuilt-pool gate. Host was not chosen
  after a result. It still lines up with the outcome (flat = mini, both take-offs = laptop): a limit, not a finding.
- Pre-onset per-update stats are close across hosts (Kasper #25285: probed ' We' share, decision grad, decision KL in
  200-update blocks). Note: a bitwise mismatch between hosts would not show a host EFFECT (sampled RL turns an ulp
  difference into a different path); only a match would rule host out.

### 10.5 Literature for the seed spread (checked 2026-11-18)
- Henderson, Islam, Bachman, Pineau, Precup, Meger, "Deep Reinforcement Learning that Matters", AAAI 2018,
  arXiv 1709.06560, PDF p.5, Figure 5: TRPO on HalfCheetah-v1, same hyperparameters, 10 seeds split into two sets of
  5: "learning curves that do not fall within the same distribution at all" (t = -9.09, p = 0.0016).
  https://arxiv.org/abs/1709.06560
- Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021,
  arXiv 2108.13264, abstract: with "a small number of runs", point estimates ignore "the statistical uncertainty";
  reliable evaluation in "the few run deep RL regime" needs interval estimates. https://arxiv.org/abs/2108.13264
  Use: credit for reporting per-seed outcomes and the seed-level interval, not a mean. One footnote at most.
- Baherwani, Chen, Qiu, Wilson, Izmailov, "Emergent Capabilities Arise Randomly from Learning Sparse Attention
  Patterns", arXiv 2606.25010 (2026-06-23); Vera's credit (#25333), checked by me in the PDF 2026-11-18 14:20 PST.
  PDF p.2 Fig. 1 caption: "models with different initialization seeds learn a repetition task at random points
  throughout training"; "Models marked with a grey X did not solve the task by the end of training". p.3: "for
  specific initialization seeds the capability never emerges" (ten Pythia seeds per scale). p.9-10 credit the
  across-seed randomness to Zhao et al. [2026]. DIFFERENCE to state if cited: their seeds differ in INITIAL WEIGHTS
  (pretraining); our three seeds start from the SAME pilot adapter and differ only in sampling and prompt order
  (live.py per-step generators + np prompt-order rng). So "seed" in our post = the random draws of one fine-tune.
  https://arxiv.org/abs/2606.25010
- DAPO / Open-Reasoner-Zero / Gao-Schulman-Hilton footnote (§4) still holds unchanged.

### 10.6 Audit of the live post (_notes/live-kl-off.md at blog b5ae308) against seeds 1-2
WRONG now: excerpt (l.6); limits 1 and the "No run used the laptop queue" line (l.19, l.100); outcome unsettled
(l.40-41) -> held; TLDR (l.52) "still did not learn"; l.76 "So the brake is in how the live run trained on its
thoughts" (2 of 3 live KL-off seeds learned; s2 at the fixed-thought update); l.86 loop-closer; l.88 "we would not
blame the KL penalty ... It went away when we replayed" (both directions unlicensed: KL-on is n=1, KL-off is 2/3);
l.82 "Only one training seed ... finished"; title (No Sign...).
STILL TRUE but seed-0-only, must be labelled so: l.66 (3,944 carrying uses, "about twice"), l.68 drift 0.135 vs
0.065, l.70 served read -7.3%, l.76 cell R numbers (replay of seed 0's thoughts), l.72 ' Let' -0.3%->0.2% (now 3
seeds: +0.5/+0.1/+2.6 points).
NOT LICENSED (keep out): "thought KL caused the live delay" (prereg if_right; KL-on comparator 1 seed, below the
2-seed control minimum); "seed lottery" as a finding (3 seeds, host confound unresolved); "the word stays learned"
(early stop at the confirming read; both take-off seeds fell after the peak: 0.847 -> 0.634, 0.628 -> 0.454).

### 10.7 Impact note for Felix (2026-11-18 14:05 PST)
The seed-spread papers are credits, not good draft targets: the only addresses are on 2017/2021 PDFs and are likely
stale (Agarwal 2108.13264 p.1 rishabhagarwal@google.com, Google Research then; Henderson 1709.06560 p.1
peter.henderson@mail.mcgill.ca, McGill then), and a 3-seed toy result gives them little. No new names from §10.
