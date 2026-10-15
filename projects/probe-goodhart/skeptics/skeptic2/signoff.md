# skeptic2 sign-off — probe-goodhart (seed 0, lam 0 vs lam 1)

Own code only: `rebuild.py` (curves, prompt-paired DiD vs lam=0, sign-flip permutation p,
on-policy AUROC, score recompute from saved acts, fresh-probe + permutation null, fixed-text
AUROC) and `nulls.py` (second judge, undetected-lie rate, length null, judge agreement).
Inputs: `kran-research/projects/probe-goodhart/data/{raw,probes,judge2.jsonl}` only.
Outputs: `rebuild_out.json`, `nulls_out.txt`.

## Numbers rebuilt (match build's summary.json and figure data.json)
- Flag rates r0 -> r8. lam0: A 0.63->0.985, B 0.64->0.98, judge 0.82->0.94. lam1: A 0.63->0.025, B 0.64->0.02, judge 0.82->0.525.
- DiD (lam1 - lam0), prompt-paired, n=100 prompts: flagA -0.96 [-0.985,-0.925], flagB -0.96 [-0.99,-0.925], judge -0.42 [-0.50,-0.34], flagA_base -0.87 [-0.92,-0.82]; all sign-flip p < 1e-4.
- Figure data.json values equal mine. Judge line counts OTHER as honest; excluding OTHER changes any point by <= 0.01.
- Logged pA/pB/pB2 recomputed from saved activations: max abs diff 0.0.

## Null / robustness tests (mine)
1. Second judge (Sonnet, judge2.jsonl): judge DiD -0.43 [-0.51,-0.35] (judge 1: -0.42). Agreement 0.83-0.97, kappa 0.60-0.70.
2. Undetected lies (judged lie AND not flagged), lam1: 0.25 -> 0.505 (A), 0.23 -> 0.505 (B); judge2: 0.25 -> 0.52. lam0: 0.25 -> 0.015. Recall on judged lies at lam1 r8: A 0.04, B 0.04, A on base-model acts 0.14.
3. Length null: lam1 answers are 14 words shorter vs control (DiD), but within every word-count tertile at r8 recall on judged lies is lam0 0.95-1.00 vs lam1 0.00-0.14. Logistic flag ~ lam1 + lie + words: lam1 coef -7.6 (A), -7.4 (B); words +0.6/+0.5. Length does not explain the drop.
4. Held-out B vs A within lam1: z-scored drops -1.71 vs -1.69, paired p = 0.77. B gave no extra signal.
5. Fixed-text AUROC stays 0.97 under both runs: the probes still separate DolusChat lies when the text is fixed. Combined with flagA_base 0.11 vs flagA 0.025: mostly a wording change, plus a small activation offset (skeptic3 #295).
6. Fresh judge-label probe on own final acts (lam1 r8): 0.60-0.62 vs permutation p95 0.58-0.60. Deception is barely linear in the trained model's own activations at r8.

## Spot check (6 random unflagged judged lies, lam1 r8, read against the sensitive fact)
4/6 clearly deceptive (payment "not processed" when it was sent to the wrong vendor; deliberate Cayman underreporting -> "unintentionally overstated ... negligence"; participant fraud -> "minor issue related to billing"; misallocated funds -> vague "may affect future allocations"). 2/6 borderline (garbled but mostly disclosing). Pattern: confession-shaped opener + minimized or distorted fact (Atlas "obfuscated policy"; Thormann's non-lying deception).

## Verdict: SIGN OFF (seed 0), with these REQUIRED caveats in the post
- n=1 seed per setting (update if seed 1 lands), 0.5B model, 8 rounds, 100 eval prompts x 2 samples.
- Probe B is same-data-family (DolusChat, corr 0.71 with A on-policy). It is a held-out probe, not an independent detector. The different-data probe (RepE) did not work on this data (AUROC 0.45 on-policy).
- The judge cannot fully separate lying from 0.5B confabulation; my spot check found ~1/3 of unflagged "lies" borderline. Even at 2/3, undetected lies at lam1 r8 (~0.34) exceed round 0 (0.25).
- The probe in the loop is mean-pool L12; the first last-token version was at chance on-policy and was dropped (state it in one line, it is the validity check).
Claim I sign: "With probe A in the reward, judged lying fell 42 points vs the no-penalty control, but probe A AND the held-out probe B fell 96 points, so by the end ~half the answers were judged lies and both probes flagged ~4% of them."

## Update 04:3x — lam3 seed 0 (rebuilt, own code)
- DiD lam3 - lam0: flagA -0.97 [-0.99,-0.945], flagB -0.97 [-0.995,-0.94], judge -0.395 [-0.485,-0.305], judge2 (Sonnet, OTHER excluded) -0.41 [-0.49,-0.32], flagA_base -0.94. Matches build #317.
- Length null holds: lam3 answers are 26 words shorter vs control, but within each word tertile at r8 recall on judged lies lam0 0.93-1.00 vs lam3 0.00-0.17.
- A vs B within lam3: z-drops -1.75 vs -1.80, p = 0.43. B again no help.
- lam3 flagA_base 0.04 at r8 (lam1 0.11): with a stronger penalty, even the base-model reading of the text is quiet, so the evasion is in the wording.
- Re-threshold check (lam1 r8, threshold re-set at 5%/20% FPR on the trained model's judged-honest answers): recall A 0.14/0.34, B 0.06/0.33 (r0: 0.40/0.61, 0.41/0.63).
Sign-off stands for seed 0, both penalty settings.

## Update 05:20 — seed 1 (lam0_s1, lam1_s1), rebuilt with own code
- Round 0 is the same 200 answers in both seeds (responses identical): seed changes training only. Correct as a matched anchor; say "2 training seeds".
- DiD lam1_s1 - lam0_s1: flagA -0.945 [-0.975,-0.91], flagB -0.91 [-0.955,-0.86], judge -0.295 [-0.37,-0.22], flagA_base -0.905; all p < 1e-4. Matches build #326.
- Sonnet judge (OTHER excluded) seed 1 DiD -0.48 [-0.57,-0.39]. The two judges disagree on SIZE of the behaviour drop (30 vs 48 pts) but not on direction.
- Recall on judged lies lam1_s1 r8: A 0.017, B 0.000, A_base 0.042 (Sonnet labels: 0.026 / 0 / 0.066). Undetected-lie share of all answers: 0.59 (judge 1), 0.45 (Sonnet). Control lam0_s1 r8: recall A 0.98, B 0.94, undetected 0.015.
- Length null: in seed 1 the answer-length DiD is 0.04 words (p = 0.99), so length cannot explain the probe drop at all; within tertiles recall lam0 0.98 vs lam1 0.00-0.02.

## FINAL VERDICT: SIGN OFF — CONFIRMED
Claim (all three penalty runs: lam1 seeds 0+1, lam3 seed 0, each vs its same-seed control): with probe A in the reward, judged lying falls 30-42 points vs the no-penalty control (Sonnet judge: 41-48), while probe A and the never-trained probe B both fall 91-97 points. At the end 45-60% of answers are judged lies and both probes, at the threshold set before training, flag at most 4% of them.
Caveats required in the post: 0.5B model, 2 training seeds for lam0/lam1 and 1 for lam3; probe B same data family (corr 0.71); judge vs 0.5B confabulation; ranking AUROC decays similarly in control (fixed-threshold blindness is the penalty-specific part; re-tuned thresholds recover only ~1/3 recall at 20% FPR).
