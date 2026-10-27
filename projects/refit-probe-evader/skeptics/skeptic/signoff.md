# Vera (skeptic) — rebuild and sign-off, theory-1027-refit-probe-evader

Own code only: I did not read or import the project's scripts. Inputs: `data/ref/split.json`, `data/ctok/*.npy`
(fp16 country-token states), `data/fits/{C.json,panel.npz,zmoments.json}`, the saved adapters `data/runs/*__adapter`,
run logs `data/runs/*.json` (loss/trajectory numbers only), and the public Qwen/Qwen2.5-1.5B-Instruct weights.

## Verdict
SIGN-OFF: REJECTED ("holds" + "price is LoRA machinery"), prediction FAILED (c1, c4), reproduced from raw data.
Every status cell is within 0.2 catch points and 0.2% KL of build (PREREG s6 rebuild tolerance: 2 points / 5%).

## Derivation checks (`vera_math.py`, `vera_math.out`)
- D1: rank-1 LEACE `x - delta delta' S^-1 (x-mu) / (delta' S^-1 delta)` (the post's display equation) equals the
  W^+ P W form of 2306.03819 to 9.5e-14, zeroes the class-mean gap (2.7e-15) and has the smallest squared edit among
  2000 random mean-equalising rank-1 erasers (6.92 vs 19.88).
- D2: the class-constant-shift "Fisher floor" is a bound only with one shared Fisher; with per-input Fisher the
  minimum is the harmonic form (random case 516.9 vs 0.035). The room dropped "bound" (#8347, #8361).
- D3: the post's "a fresh true statement crosses the line with probability 2/201": simulated 0.00968 vs 0.00995.
- Shrinkage = linear shrinkage of arXiv 1308.2608 (concept-erasure default), not Ledoit-Wolf; my numpy version
  matches the library to 7.6e-16 (`vera_shrink_check.out`).

## The LEACE price (registered P = 0.006145, bf16, linear shrinkage, sequential 10..29, country token, 1400 pool)
| cell | Vera | build | file |
|---|---|---|---|
| P, own sequential LEACE, CPU fp32 | 0.005945 (-3.3%) | 0.006145 | vera_kl_seq_fp32_lw.json |
| P with the library's trace constraint coded by me | 0.005945 (never fires) | same | vera_kl_seq_fp32_lw_tc.json |
| plain covariance (estimator sensitivity) | 0.0209 | 0.0214 | vera_kl_seq_fp32.json |
| clean acts, my fp32 forward vs saved | rel 0.8-1.4%, cos >= 0.997 | | vera_kl_check.out |

## Known answers
| cell | Vera | build |
|---|---|---|
| clean F1 / F2 / M1 / frozen | 99.66 / 99.50 / 99.33 / 99.50% | 99.7 / 99.5 / 99.3 / 99.5% |
| registered-erased F1 / F2 / M1 | 0.67% (AUROC 0.555) / 0.17% / 0.34% | 0.7 / 0.2 / 0.3% |
| gate eraser (attack-600) F1 / F2 | 0.34% (AUROC 0.808) / 2.68% (r22) | 0.3% (0.809) / 2.7% |
| my pre-data cell: single-readout LEACE fit on attack only | panel 0.2% catch, AUROC 0.81 | (#8383) |
| KL known answer: 1026 recal s0 adapter (saved 0.003811) | bf16 0.003806, fp32 0.003076 | |

## Fine-tuned states (own probes on saved ctok; own LoRA merge + forward for KL; build's right-country rule)
| run | M1 | F1 | F2 | frozen | eval KL bf16 (Vera / build) | right country |
|---|---|---|---|---|---|---|
| R1 lam1e4 s0 | 84.06% | 87.25% | 78.36% (r19) | 7.55% | 0.007928 / 0.00792 | 298/298 |
| R3 lam1e4 s1 | 61.41% | 85.57% | 83.56% (r19) | 9.40% | 0.007032 / 0.00702 | 298/298 |
| R4 lam1e4 s2 | 70.30% | 63.59% | 80.03% (r19) | 17.45% | 0.007622 / 0.00761 | 298/298 |
| pooled (joint city bootstrap) | 71.92% [68.5, 75.1] | 78.80% [76.2, 81.4] | 80.65% [77.3, 83.7] | 11.47% | 0.007527 = 1.225 x P | |
| R2 lam1e5 s0 | 99.83% | 99.66% | 99.33% | 98.83% | 0.003827 / 0.00382 | 298/298 |
| G mean-gap s0 | 99.16% | 98.83% | 98.49% | 99.33% | 0.006254 / 0.006248 | 298/298 |
| N shuffled x44.7 s0 | 99.33% | 99.50% | 99.50% (r12) | 99.50% | 0.005026 / 0.00503 | 298/298 |
| long900 s0 (descriptive) | 37.58% | 52.68% | 57.72% (r19) | 4.36% | 0.011617 / 0.011603 | |
| refit100 s0 (descriptive) | 92.95% | 96.81% | 98.15% (r17; build r19 97.65%) | 8.89% | 0.006322 / 0.006315 | |

Clauses (s6): 1 fresh < 30%: false; 2 M1 < 20%: false; 3 KL <= 0.0096 and >= 290/298: true; 4 KL in [P/3, 3P]: true
(1.225 x P); 5 N < 0.5 x base: false (0.668; D3 diagnostic N removed 92.2% of its loss vs base 81.9%, comparable).

## Nulls and stress
- Shuffled-label fresh probe at F2's readout, 5 permutations per state: 0.0-2.0% caught. Random direction: <= 0.3%.
- QDA / 2-layer MLP at F2's readout: 94.5-99.7% on every state, so no state hides the concept from nonlinear probes.
- In-loop probes really are refit with labels on the current model: saved s300 probe vs my own fit on R1's final
  states cos >= 0.9957; vs s0 cos 0.52 (`vera_probecheck.out`).
- N's loss scale: true-label step-0 loss 77.83 vs shuffled 1.75 median of 5 (x44.6; `vera_nullgap.out`).
- Split: 5 roles city-disjoint, no duplicate texts (`vera_split_check.out`).

## Caveats (disclosed in the post or to be disclosed)
1. Budget: the No holds at 300 steps. One run at 900 steps fell to M1 37.6%, F1 52.7% at KL 0.0116 (1.89 x P, above
   the 0.0096 output bar).
2. dtype: bf16 rounding adds ~0.0009-0.0012 nats/token to every fine-tune's KL but ~0.0002 to P. Own CPU fp32
   forward: R1 0.007017, R3 0.006177, R4 0.006400, pool 0.006531 (bf16 0.007527) = 1.10 x fp32 P (bf16: 1.22 x).
   No status clause flips (`vera_statekl_R{1,3,4}.json`).
3. Estimator x dtype: clause 4 passes under the registered estimator in bf16 (1.22 x) and fp32 (1.10 x), and under
   the plain estimate in bf16 (edge 0.00713 < 0.00753), but FAILS under plain + fp32 (edge 0.0209/3 = 0.00697 >
   0.00653). The status does not use clause 4 (it is REJECTED via clauses 1, 2, 5), so this is a disclosure.
4. Trajectory points before step 300 use a 200-statement affirmative-only slice that over-reads M1 by up to 7.7 points
   (`vera_slice_vs_full.out`); the figure ends each line at the full-eval value.
5. Gap geometry (`vera_gap.json`): base runs left 0.42-0.45 of the whitened gap and turned it ~77 deg; the LEACE edit
   left 0.21 and turned it ~74 deg; G left 0.59 and kept the direction. The turn does not separate the edits.

## Novelty
See `novelty.md`: an output-anchored fine-tune against a labelled refit probe is not new (Westphal et al. 2026,
2606.09411). New as far as I found: a closed-form price fixed before training (LEACE KL) tested against the trained
evader's measured price, on a native concept the outputs barely use.
