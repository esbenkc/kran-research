# Vera (skeptic) sign-off — oracle-1018-probe-ensemble

Title: "Does training against many lie detectors at once cut lying twice as much as training against one?"
Answer I reproduce: **NO (REJECTED)** under the frozen PREREG.md rule (section 5), primary judge
gemini-3-flash-preview (DEVIATIONS D1).

All numbers below come from my own code reading only the saved rows in
`projects/probe-ensemble/data/raw/eval_<run>.jsonl` (identical to `data/results/` except the added
`judge_model` stamp on lam1_s0; 0 differing shared fields).

## Key numbers (vera_check.py, 10,000-draw paired prompt bootstrap, numpy seed 0) — vera_check_judge1.txt

| seed | single r0 -> r8 | cut (95% CI) | ensemble r0 -> r8 | cut (95% CI) | ratio (95% CI) | P(boot ratio >= 2) |
|---|---|---|---|---|---|---|
| 0 | 0.780 -> 0.555 | 0.225 (0.14-0.31) | 0.780 -> 0.515 | 0.265 (0.185-0.345) | 1.18 (0.83-1.79) | 0.009 |
| 1 | 0.780 -> 0.485 | 0.295 (0.205-0.385) | 0.780 -> 0.490 | 0.290 (0.20-0.38) | 0.98 (0.67-1.42) | 0.000 |

Guard: single cut >= 0.10 and CI excludes 0 in both seeds. Ratio < 2 and CI upper < 2 in both seeds -> REJECTED.
Matches Bruno's analysis.py (s0 1.18, CI 0.83-1.81; s1 0.98, CI 0.68-1.42; his 2,000 draws).

## Null and robustness tests
- Paired sign-flip permutation of arm labels per prompt at round 8: single-minus-ensemble gap 0.04 (p=0.44)
  and -0.005 (p=1.0). The two arms are indistinguishable at round 8.
- Seed swap: ensemble s1 / single s0 = 1.29; ensemble s0 / single s1 = 0.90; pooled 1.07 (vera_rounds_judge1.txt).
- Round choice: per-round ratios are unstable (seed 1: 2.67 at r2, 2.87 at r4; seed 0: 0.06, 0.12). The
  mean cut over rounds 2-8 gives 0.59 and 1.72: both < 2. The r8 endpoint was pre-registered; the post must
  show all rounds because seed 1's ensemble dipped to 0.35 at r4 and came back to 0.49.
- Second judge, other family, my own rubric (claude-haiku-4-5, vera_judge2.py; 0.94 on 200 DolusChat
  reference replies, vera_judge2_validate.py): cuts single 0.17 / 0.19, ensemble 0.21 / 0.275, ratio 1.24
  (CI 0.84-1.88) and 1.45 (CI 0.95-2.39). Point ratios < 2 in both seeds, but seed 1's CI upper is 2.39,
  so under this judge the strict REJECTED rule would read UNCONFIRMED for seed 1. The post should state the
  ratio range (about 1.0-1.2 primary, 1.2-1.45 second judge), not "no more than one probe".
- Judge provenance: rows of 3 runs carry judge_model=gemini-3-flash-preview; lam1_s0 started before the
  stamp. The repo's lib/llm cache holds 0 gpt-4.1-mini entries and only gemini-3-flash-preview entries
  written after 03:14 CEST; OpenAI had no credit from 03:08 (429). Round-0 answers are byte-identical
  in all 4 runs (200/200) with identical labels.

## Held-out attention probe (frozen; vera_auroc.py, OTHER dropped) — vera_auroc_pAttn.txt
- r0 0.787 (CI 0.705-0.862). r8: single 0.735 / 0.705, ensemble 0.677 (CI 0.600-0.750) / 0.732 (CI 0.649-0.803).
  Shuffled-label 99th percentile 0.58-0.62: the probe still beats chance at r8. Seeds split around 0.70,
  so the prediction's AUROC half is unsettled; no consistent ensemble-vs-single difference.
- Flags at the 5%-FPR thresholds (vera_members.py): any of 8 members 0.835 at r0 -> single 0.20 / 0.27,
  ensemble 0.235 / 0.055; every member under the single penalty 0.04-0.17 at r8; attention probe
  0.59 -> 0.095 / 0.12 (single), 0.075 / 0.02 (ensemble). Judged lying stays ~0.5.

## Probe fit (vera_probes.py, vera_probes_out.txt)
All 8 member AUROCs, thresholds, ensemble max and attention AUROC reproduce to 3-4 dp. Ensemble
honest-vs-deceptive margin: 0.606 vs probe A 0.794 on calib; on-policy r0 0.199 vs 0.374 (vera_auroc.py).

## Definitions checked
- Silenced, not blinded (vera_member_auroc.py): under the one-probe penalty the eight-probe max still ranks
  deceptive above truthful answers with AUROC 0.763 / 0.720 at r8 (0.797 at r0); member mean 0.742 / 0.694
  (0.772 at r0); its deceptive-minus-truthful gap at r8 is +0.29 / +0.27 (r0 +0.20). The flag drop comes from
  scores shifting below fixed thresholds. "Fooled all eight" / "nothing left to catch" overstate this.
- Ratio: cut = own r0 - r8 per arm per seed; rate = DECEPTIVE / 200 answers (OTHER counts as not deceptive,
  as in PREREG s4); ratio CI paired over 100 prompts. No stretch in the ratio. Not run: control arm (D3), so
  no control-adjusted cut; the held-out AUROC has no lambda=0 drift reference.

API spend (mine): ~$1.93 (claude-haiku-4-5, ~5,070 calls, cached).
