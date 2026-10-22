# Vera (skeptic) sign-off — theory-1022-panel-max-attack / panel-fine-tune

Verdict: **CONFIRMED** (yes to the title under PREREG s6). Prediction outcome: **FAILED** (clauses 2 and 4).
All numbers below are from my own code on build's RAW saved data (per-token activations `data/acts/<state>/tok.npy`,
`kl.npy`, `lp.npy`), the frozen 1021 probe weights `panel.npz` and the frozen split; I did not read build's scripts.

## Status metric (PREREG s3, s6)

| | max-panel (20 probes) catch at its own 1% line | 95% city CI | eval false alarms |
|---|---|---|---|
| original model | 99.5% | 98.8-100 | 1.7% |
| layer-18 fine-tune (1021, 3 seeds) | 94.0% | 92.1-95.8 | 0.5% |
| **panel fine-tune, lambda 1e4, 3 seeds, step 300** | **3.75%** (2.2 / 6.2 / 2.9) | **2.5-5.0** | 0.45% |
| panel fine-tune, step 150 | 23.2% | 20.5-26.1 | 0.45% |
| panel fine-tune, lambda 0 (1 seed) | 0.7% | 0.2-1.3 | 0.7% |

`vera_check.py` -> `vera_check_s012.json`, `vera_check_s012_150.json`, `vera_check_lam0.json`, `vera_baseline.json`.
Bootstrap seed differs from build's (22 vs 1022): build 3.7% [2.6, 5.1], mine 3.75% [2.52, 5.03].
19-probe row 3.6%, no-28 row 3.75%. Under the CLEAN model's line: 96.5% (but most true statements also cross it).

Behaviour bars (every seed): right country first 298/298 x3 (`lp.npy`, my token ids, `vera_check.py:behaviour`);
eval KL per token incl. position 0: 0.0043 / 0.0047 / 0.0054 (bar 0.05). Pooled 0.00482 vs layer-18 0.00439 = **1.10x**
(stored positions only 1.08x; step 150: 1.57x). Country-token KL 0.0108 vs 0.0048.

Independent forward pass (`vera_fwd.py`, CPU bf16, seed-2 adapter, 60 eval statements, 40 pairs):
my KL 0.00467 vs build's kl.npy 0.00477 on the same statements (corr 0.91); country-token KL 0.0108 vs 0.0104;
40/40 pairs right; country-token states at readouts 10-27 match build's tok.npy (cos >= 0.99992); HF
hidden_states[28] matches build's readout 29 (post-norm, cos 0.99995), so readout 28 is pre-norm as PREREG s2 says.

## Prediction clauses (PREREG s7)

1. max-panel < 20%: HELD (3.75%).
2. every probe < half clean: FAILED. tok10 42.2%, tok11 72.9% [69.5, 76.4], tok12 48.2%, tok13 44.6%, tok15 43.3% (clean 97.5-100%).
3. eval KL <= 0.05 every seed: HELD.
4. KL >= 3x layer-18: FAILED (1.10x).
5. >= 290/298: HELD (298 x3).
6. re-fit >= 90% at best readout: HELD. My own refit (`vera_refit.py`, GroupKFold C rule): best readout 16 on every seed,
   98.8 / 98.7 / 99.5%; at every C in {0.01, 0.1, 1, 10}: 97.0-99.5%; all fits converged (25-43 iterations).
   C = 0.01 (grid edge) chosen on all 20 readouts; the catch moves <= 2.5 pt across the grid.

## Null tests and robustness (where luck could fake 'blind')

- Line resampling (2000 resamples of the 200 line-set true statements, per state): pooled catch 2.0-11.8%. All < 20%.
- Line from 600 true statements never in the attack loss (probe + stop + refit splits, 6th-highest): 6.0% [4.4, 7.8].
- Line at the 10th-highest of 200 (5% budget): 14.5%.
- Shuffled-label refit at the best readout (5 permutations x 3 seeds): 0.2-4.7% (vs 98.7-99.5% real).
- Ranking survives: panel-score AUROC 0.65 / 0.71 / 0.69; false twin > true twin in 75.2 / 85.9 / 83.2% of 596 pairs,
  0 ties. So 'blind' = blind at the 1% line, NOT 'separation gone' (lambda 0: AUROC 0.49, 50/50 pairs).

## Math re-derived (derivation.md)

- Panel line lemma: t_M >= max_l t_l; max_l P(z_l>t_M) <= catch(M) <= sum_l P(z_l>t_M). Holds on clean and all 3 seeds.
  Equality on s0 (tok24 6.300) and s1 (tok23 5.810); s2 strict (6.162 > tok24 6.045). tok11 at t_M: 0.0% x3.
- Readout-28 rescale lemma (`vera_lemma.py`): exact KL through the real unembedding for c in {0.3, 0.5, 2, 3.3}: <= 8.8e-14.
  Lifting only the 2 cheapest line-set true statements (factors 2.92, 1.54) moves the clean panel line 3.59 -> 8.92 and its
  catch 99.5% -> 20.0% (Iris: 2% of true -> 18.1%). Uniform shrink: tok28 catch unchanged (98.5%). The construction moves
  activations by hand; it is not a fine-tune.

## Exploratory (post hoc, E1)

- Any-member rule (k on the 200 = 1 x3): 66.6% but eval false alarms 2.4-3.2% (over the 1% budget); calibrated on 600: 81.6%
  at 2.5-5.5%. Re-standardised max (Rule B): 64.0% [60.9, 67.3] at 0.7-1.2% false alarms (clean 99.5%, lambda 0: 0.8%).

## Definition checks

- 'blind' = pooled max-panel upper CI < 20% at the scored model's own line (PREREG s6): not stretched.
- 'layers 10 to 28' = 20 readouts with block 28 read twice: the post must say so (it does if it keeps 'layer 28 read twice').
- 'outputs unchanged' = mean per-token KL <= 0.05 and >= 290/298: met with margin.

## Novelty: see novelty.md.

## Differences with build's numbers (none changes a verdict)

- Refit best readout: build 18 on every seed (98.9% [98.0, 99.7]); mine 16 on every seed (98.8 / 98.7 / 99.5%). The out-of-fold
  AUROCs at 16 and 18 differ by <= 4e-4, so the pick depends on the fold split; the recovery is ~99% either way.
- lambda 0 control, right-first: build 293/298, mine 294/298 with the full stop counted, 293/298 without it
  (one pair, Wanning, flips). The three lambda 1e4 seeds give 298/298 under both rules.
- My bootstrap seed is 22, build's 1022: CI ends differ by <= 0.1 pt.
