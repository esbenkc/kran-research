# Vera (skeptic) — sign-off, theory-1018-shift-spread

**Verdict: CONFIRMED** on the pre-registered primary, with three conditions on the wording (below).

Data rebuilt: kran-research `faa54c5`, `projects/rethreshold-probe-evasion/data/{clean_acts.npz, runs/lam*__seed*.npz}`
(per-example frozen-probe scores of all 2992 statements at step 0 and every 10 steps). No build code used.
Definitions: PREREG.md frozen at `beff840` (2026-10-18 15:42:02 CEST).

## Primary (my numbers, `vera_rebuild.py` → `vera_rebuild_full.json`)

| retain weight λ | per-seed share | pooled share | paired 2000-draw 95% CI | AUROC drop | common-mode c |
|---|---|---|---|---|---|
| 1e5 (headline, frozen rule) | 0.856 / 0.940 / 0.919 | **0.904** | **[0.887, 1.002]** | 0.0002 | 0.90 / 0.95 / 0.96 |
| 3.16e4 | 0.727 / 0.783 / 0.783 | 0.765 | [0.631, 0.986] | 0.0013 | 0.92 / 0.96 / 0.97 |
| 1e4 | 0.498 / 0.670 / 0.675 | 0.614 | [0.463, 0.962] | 0.0030 | 0.91 / 0.94 / 0.94 |
| 0 (no retain term) | 0.007 / 0.009 / 0.009 | 0.008 | [0.002, 0.028] | 0.423 (AUROC 0.999 → 0.57-0.58, not inverted) | 0.88 |

- Headline λ = largest grid value with all 3 seeds ≥ 50-point drop at t0 = 1e5 (all four values qualify).
- Status rule (PREREG): lower bound 0.887 > 0.5 → CONFIRMED. Prereg outcome: 0.904 ≥ 0.5 and AUROC drop 0.0002 ≤ 0.15 → held.
- Every number matches Bruno's `results.json` to ≥ 4 decimals (max |Δ share| 5e-5 over 12 runs vs `figures/.../data.json`).
- Independent-per-run bootstrap would give [0.900, 0.994]: narrower, as predicted (#2717). The paired one is reported.
- Step-0 scores of every run equal the clean scores exactly; saved final activations reproduce saved scores (max |Δ| 0.002, fp16).
- Splits: 748 cities, none in two splits; refit-200 negatives never seen by the attacker (`vera_gate.py`).

## Nulls and robustness (my code)

1. **Calibration-set luck** (resplit null, 2000 redraws of 200 from the 796 held-out clean negatives): at λ=1e5 mean 0.975,
   95% range 0.922–1.013; the frozen refit-200 is at the very bottom. Cause: the frozen refit-200 holds the two highest of all
   796 attacked negatives (ranks 1 and 2 at λ = 1e4 and 1e5; `vera_lam0_probe.out`). So luck works AGAINST the claim; the
   headline understates recovery. Refit-set size: n = 100 / 200 / 400 → 0.985 [0.90, 1.09] / 0.975 [0.90, 1.01] / 0.973 [0.94, 1.00].
2. **Seeds**: all 3 headline seeds sit above 0.5 (0.856-0.940).
3. **λ=0 arm** (strongest attacker): recovery 0.008. A fresh logistic probe on the λ=0 activations still ranks the classes
   (eval AUROC 0.980 / 0.980 / 0.984) but reaches only 0.06 / 0.20 / 0.12 TPR at 1% FPR (`vera_lam0_probe.out`).
   The signal is NOT gone; it moved off the frozen direction and lost its low-FPR margin.
4. **Clean binormal validity**: binormal TPR at t0 0.968 vs empirical 0.960 (miss 0.8 pts < 5) → allowed as a theory line.

## Theory (my derivation `derivation.md`, checks `vera_theory_check.py`, `vera_theory_test.py`)
Re-derived E6 share(D,c,p) and its limits (p→0: c; p→∞ with c<1: 0; c=1: 1). E6 vs measured share over 171 checks:
MAE 0.184 overall; 0.084 (1e5), 0.203 (3.16e4), 0.373 (1e4), 0.056 (0). E6 predicts ≈0.99 for every retain-constrained run
and misses the middle of the trend. The miss is threshold placement by a few hard true statements: top-decile negatives move
4-6% less than mid-decile ones in all 9 retain runs (top/mid ratio 0.936-0.961), and the frozen set holds the top 2.

## Conditions on the post (a FAIL of the post if missing)
1. The TLDR states the λ=0 result next to the 90%: re-fitting recovers 90% for the most constrained attacker, 61-76% for
   harder ones, and about 1% when the fine-tune has no retain term. Output change is tiny in all arms (held-out KL 0.002 vs 0.007).
2. Do not write that the λ=0 attack "removed" or "erased" the information: a fresh probe still has AUROC 0.98.
   Say it removed the margin the old probe needs at 1% false alarms.
3. Novelty: Libon et al. 2609.38645 App. B.5 already saw a frozen probe evaded by a translation; Bailey et al. 2412.09565
   Sec 4.2 retrain from scratch. The post claims only the measured share and its fall with push. Also disclose that the
   "clean negatives" are held-out TRUE city statements of the same template, not generic clean traffic.
