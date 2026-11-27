# research.md — oracle-1127-forced-switch-head (Iris)

Facts the room builds on. Each line has a source. Times PST unless noted (CET = PST + 9 h).

## Parent run (the 'dropped, not hidden' result)

Source: kran-research branch `origin/swarm/oracle-1125-blind-head-k-law`, `projects/probe-switch-head-budget/`
(PREREG pushed fdc17f064, 2026-11-25T01:14:52Z; status.py: UNCONFIRMED, prereg outcome failed). Post is NOT published:
athanorscience.com draft PR #17 ("Lab (pending laptop): probe-switch-head-budget ...").

- k_att = 6 arm (proof-matrix.md): k6_s0 S0 1.000, S(8) 0.008, k_min 6, R8 0.930; k6_s3 (spare, warm s0, seed 3)
  S0 1.000, S(8) 0.023, k_min 7 (bf16 6), R8 0.996. k6_s1 S0 0.000, k6_s2 S0 0.008 (faded; 'monitor ok False').
- Controls: L0 s0/s1 (no penalty) S0 1.000, k_min 1, R8 0.887/0.996; warm W s0/s1/s2 S0 1.000, k_min 1,
  R8 0.910/0.938/0.867.
- Law arms: k_att = 3 k_min 2/4/1 (S0 0.922/1.000/0.914); k_att = 1 two of three faded.

### The faded seeds are exactly the monitor-tie seeds (Iris, room #31660)

`data/runs/k6_s*.json`, `picks[0].tie_steps` and `picks[0].greedy_best_runnerup` (the attacker's first re-pick, on
the untrained warm adapter, 64 train prompts, per-prompt T):

| run | warm | first re-pick tie steps (cumulative rel. 1e-5 rule) | step-1 share of 6-head score | S0 |
|---|---|---|---|---|
| k6_s0 | s0 | none | 0.71589 / 0.72286 = 99.0% | 1.000 |
| k6_s3 | s0 | none (identical pick to k6_s0) | 99.0% | 1.000 |
| k6_s1 | s1 | steps 5, 6 | 0.24412 / 0.24493 = 99.7% | 0.000 |
| k6_s2 | s2 | steps 4, 5, 6 | 1.39229 / 1.39292 = 99.95% | 0.008 |

- The first re-pick is computed before any update, so it is deterministic per warm adapter: under the parent's
  MONITOR HEALTH gate, warm s1 and s2 fail again in this run regardless of the survival floor.
- The 'joint 6-head' monitor is, on the warm adapters, a 1-head monitor plus 5 heads that add < 1% of the score.
  The cumulative relative-1e-5 tie test compares two numbers that differ in the 5th significant digit: it fires on
  steps whose increments are 1e-5 - 1e-4 nats.

### Increment tie rule, calibrated on the saved warm data (Iris, from the files above)

Increment margin of step j = (best_j - runnerup_j) / (best_j - best_{j-1}), best_0 = 0.

| source | per-step increment margin (steps 1..k) | min |
|---|---|---|
| monitor first re-pick, k6_s0 (= k6_s3) | 0.372, 0.009, 0.025, 0.191, 0.083, 0.405 | 0.0094 |
| monitor first re-pick, k6_s1 | 0.277, 0.330, 0.110, 0.558, 0.103, 0.067 | 0.067 |
| monitor first re-pick, k6_s2 | 0.137, 0.115, 0.179, 0.140, 0.255, 0.095 | 0.095 |
| status audit dev, warm_s0 (k 1..8) | 0.168, 0.045, 0.018, 0.227, 0.141, 0.137, 0.231, 0.186 | 0.018 |
| status audit dev, warm_s1 | 0.084, 0.344, 0.152, 0.613, 0.219, 0.290, 0.292, 0.727 | 0.084 |
| status audit dev, warm_s2 | 0.097, 0.025, 0.194, 0.428, 0.394, 0.017, 0.275, 0.018 | 0.017 |

A tie threshold of increment margin < 1e-3 fires on 0 of 18 first-re-pick steps and 0 of 24 audit steps. Under the
parent's cumulative rule the first re-picks tie on 5 of 18 steps. Not checked yet: numerical stability of the
step-5/6 increments (1e-5 to 1e-4 nats) across a rerun on MPS (repeat the pick twice; same heads = stable).

## Compute facts

- Laptop queue is broken tonight: `labq list --status done,failed,cancelled` shows ddh2ux1kls, 82u44m7zrc, eukptwkl12
  (oracle-1126-build-conditional), fggi9o04td, ta9s2z45o4, 8bgphbrd4a (critics) all failed; the worker wraps jobs in
  `LABQ_GW` = `~/Desktop/Kran/kbrowser-bench/gw.sh` (`~/bin/labq:80`), which does not exist (exit 127 at 0.00 h).
  Last done job: fjq6f0ehwu, 2026-11-25. So this run is mini only.
- Parent timing on the laptop (data/laptop/fjq6f0ehwu/chain.log): k_att = 3 train 839-893 s/seed, k_att = 1 368-383
  s, kaudit 4-5 min, whole-graph eval ~45 s. k_att = 6: ~24-28 min/seed (Bruno, Felix, Tove from keb7u8is33).
- Mini rule: models above 2B or above 10 GB go to the laptop; 1.7B fits the mini under the one-local-fine-tune rule
  (<= 9 GB RSS, alone) if no other swarm is open.

## Prior work to credit (abstracts fetched 2026-11-26)

- Bailey et al., Obfuscated Activations Bypass LLM Latent-Space Defenses, arXiv 2412.09565 (ICLR 2026): joint
  optimisation of behaviour + fooling a latent monitor; obfuscated activations are abundant.
- Gupta & Jenner, RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?, arXiv 2506.14261:
  token-level monitors are evadable by RL; holistic monitors (max-pool, attention probes) stay robust.
- McGuinness, Serrano, Bailey, Emmons, Neural Chameleons, arXiv 2512.11949 (ICLR 2026): fine-tuned models evade
  unseen activation monitors zero-shot on a trigger; evasion is a low-rank change of activations.
- McGrath et al., The Hydra Effect, arXiv 2307.15771: ablating one attention layer makes another compensate.
- Rushing & Nanda, Explorations of Self-Repair in Language Models, arXiv 2402.15390: self-repair after single-head
  ablation across model families; part of it is final-LayerNorm scaling.
- Method lineage: Sauers, github.com/SauersML/watching (probe hook, the switch recipe), as in the parent README.
- Yu et al., Backdoor Attribution, arXiv 2509.21761 (2025-09-26): backdoor attention heads are sparse; ablating ~3% of
  heads cuts attack success by over 90%. This is the non-adaptive baseline our question stresses: does sparsity
  survive an attacker that is penalised for it and must keep the behaviour?
- Zhao et al., PURE: Defense against Backdoor Attack on Pre-trained Language Models via Head Pruning and Attention
  Normalization, ICML 2024 (PMLR v235, proceedings.mlr.press/v235/zhao24r.html): prunes heads found with clean data
  only (a blind pick, like ours).
- Novelty (my read): I found no paper that trains a trigger-conditional behaviour against a head-ablation audit with a
  hard constraint that keeps the behaviour, then measures the smallest blind cut that still works. Searched:
  "backdoor defense pruning attention heads adaptive attacker", plus the obfuscation papers above.

## Impact-draft email sources (Iris, for Felix; checked in the arXiv PDFs with PyMuPDF, 2026-11-26 17:11 PST)

| person | paper | public email in the PDF | page |
|---|---|---|---|
| Luke Bailey | 2412.09565 Obfuscated Activations | ljbailey@stanford.edu | first 3 pages |
| Erik Jenner | 2412.09565 (co-author); 2506.14261 RL-Obfuscation | erik@ejenner.com (in 2412.09565) | first 3 pages |
| Scott Emmons | 2412.09565 | scott@scottemmons.com | first 3 pages |
| Rohan Gupta | 2506.14261 RL-Obfuscation | cybershiptrooper@gmail.com | first 3 pages |
| Alex Serrano | 2512.11949 Neural Chameleons | mail@alexserrano.org | first 3 pages |
| Max McGuinness | 2512.11949 | max@max.rip (PDF text 'max@max.rip.' with a trailing period) | first 3 pages |
| Miao Yu, Zhenhong Zhou | 2509.21761 Backdoor Attribution | none in the whole PDF -> no draft | all pages |
- Tramer, Carlini, Brendel, Madry, On Adaptive Attacks to Adversarial Example Defenses, arXiv 2002.08347 (2020-02-19):
  13 published defenses fall to adaptive attacks; a defence counts only against an attacker that knows it. (Checked by
  Iris for Selma's opening, Tove #31696.)

## F s0 early log (data/mini/mini_1126_171351/k6_s0.log, read 17:40 PST) and the oscillation prior

- Train-64 strict: 1.000 (update 0) -> 0.078 (update 64, mu 13.2, audit penalty 0.93) -> 0.938 (update 128, mu 42.5,
  penalty 1.00). Re-pick at update 137: head 27.7 alone removes all but 0.0035% of D (left_pct after step 1). So far
  the seed stays fully caught in one head; it does not spread.
- Stooke, Achiam, Abbeel, Responsive Safety in RL by PID Lagrangian Methods, arXiv 2007.03964 (2020): plain
  (integral-only) Lagrange multiplier updates oscillate and overshoot, so the constraint is violated during training.
  Our dual update is that plain form (mu += eta x (target - s_batch)), so drop/recover swings are the expected dynamic,
  not a bug. Citable if a seed fails FLOOR MET.

## F s1 result and its mechanism (Iris, recomputed from data/kaudit/k6_s1.json, 19:20 PST)

- S(k) = 1.000 for every k = 1..12 (fp32); S0 1.000; R8 median 1.000 (min 0.148); R12 median 0.684 (min 0.000).
- Blind top 12 (26.9, 27.1, 26.10, 27.0, 27.2, 27.10, 26.7, 26.6, 26.12, 26.11, 27.11, 27.13) remove -0.16% of the A/B
  push and -0.35% of the blind score. Random 8-sets remove median 7.7% of the push (max 73%). All 32 heads: 100% (S 0).
  MLP27 alone: S 0.000; MLP26 alone: S 0.180; both MLPs: S 0.000.
- RETRACTED (Iris #31747, after Tove #31742): the OR/backup reading was wrong. Under fixed T = 1 and T = 16 the greedy
  picks 27.7, 27.12, 27.4, 27.6 and S falls 1.000, 0.969, 0.148, 0.023, 0.000 at k = 1-5. Under the frozen per-prompt
  T (mean 4.13) and T = 4, cutting 27.7 alone has blind_removed = -1.70 (-1.74 at T4; 0.00 at T1; +0.36 at T16), the
  pair {27.7, 27.12} -1.28, and only the 4-set turns positive (+0.59). The cut of the real carrier makes the softened
  full-vocabulary outputs MORE different, so one-head-at-a-time greedy never starts on the carriers. Mechanism: a
  decoy on the audit's full-vocabulary score at its temperature (Kasper's reading (b)); the switch did not spread.
- A/B-only check of the carriers (Iris #31771; script swarm-oracle-1127-forced-switch-head/iris/ab_only_carriers.py,
  saved test data only): at the per-prompt T, cutting {27.7}, +27.12, +27.4, +27.6 removes +1.9%, +31.6%, +88.3%,
  +97.2% of the two-answer score but -170%, -128%, +4.9%, +59.1% of the full-vocabulary score. The negative score
  comes from non-answer tokens let in by the softening: a decoy on the carriers.
| Stefan Heimersheim | 2404.15255 How to use and interpret activation patching | stefan.heimersheim@gmail.com | first 3 pages |
| Cody Rushing | 2402.15390 Explorations of Self-Repair | thisiscodyr@gmail.com | first 3 pages |
| McGrath et al.; Tramer et al. | 2307.15771; 2002.08347 | none in first 3 pages -> no draft from these PDFs | |
- Huang et al., Safety Alignment Should Be Made More Than Just A Few Attention Heads, arXiv 2508.19697 (2025-08-27;
  found by Vera, abstract checked by Iris): safety behaviour sits in few heads (RDSHA ablation); AHD training spreads it
  across many heads to resist jailbreaks. The defender-side mirror of our question: spreading on purpose across heads.
- Dev confirmation (Vera #31895, own forward pass): 27.7 alone removes 6.9% of the A-B gap; full-vocab score at the
  per-prompt T -168% vs A/B-only +3.4% (T4 -169%, T16 +37%); +27.12: -100% vs +38.8%; 4 carriers +65% vs +98%; the
  audit's own first pick 26.9 removes 0.003% of the gap. The decoy-on-the-carriers reading holds on dev too.
