# Vera (skeptic) sign-off — theory-1024-curvature / curvature-cut-probe

**Verdict: REJECTED (no to the title), CONFIRMED by my rebuild** under PREREG s7 + DEVIATIONS D1-D11.
**Prediction outcome: FAILED.**

Every number below comes from my own code (this folder) on build's raw saved files: `data/runs/<arm>__lam10000__seed<s>/ckpt_{0,50,100,150}.npz`
(probe scores of all 24 probes, per-position KL with `koff`/`cpos`), `acts_150.npy`, `data/clean/z_post.npy`, `data/fits/probes.npz`,
`data/ref/split.json`, and the HF safetensors of Qwen2.5-1.5B-Instruct for the unembedding. I did not read or reuse build's analysis code.

## 1. Headline: PRICE (country-token KL, D1 rule, horizon 150, `vera_price.py`, `vera_price_H150.json`)

| arm | seed 0 | seed 1 | pooled PRICE | first-crossing | step-150 re-set catch | step-150 clean-line catch | all-token KL @150 |
|---|---|---|---|---|---|---|---|
| full probe U (new code) | 0.00500 | 0.00586 | **0.00543** | 0.00965 | 1.17 / 1.51% | 66.4 / 61.4% | 0.00520 / 0.00457 |
| cut probe P (k=128 refit) | 0.00531 | 0.00434 | **0.00483** | 0.00697 | 1.17 / 0.67% | 67.0 / 68.8% | 0.00660 / 0.00382 |
| random-cut R (k=128 refit) | 0.00260 | 0.00278 | **0.00269** | 0.00411 | 0.34 / 0.67% | 27.0 / 35.7% | 0.00433 / 0.00344 |

P/U = **0.889**, R/U = **0.495**. Max-seed P 0.00531 <= 1.5 x max-seed U 0.00879. PREREG s7: PRICE(P) <= 1.5 PRICE(U) AND max-seed rule -> **NO**.
(The YES branch also fails: P/U < 3 and P fell to 0.9% pooled catch.) Same numbers as Bruno #6372, Hugo, Nadia, Felix's exporter.

Sensitivity of the answer to the price definition (same data): first-crossing P/U 0.72; all-token KL at step 150 P/U 1.07; every single-seed
pairing P/U 0.74 / 0.87 / 0.91 / 1.06 (`vera_boot.json`). No reading reaches 1.5, none is near 3.

## 2. Nulls and controls (my code)

| test | result | artifact |
|---|---|---|
| statement luck: city bootstrap, 298 cities, 2000 draws | P/U 95% [0.863, 0.915], R/U [0.478, 0.515]; P(P/U >= 1.5) = 0/2000 | `vera_boot.py`, `vera_boot.json` |
| seed luck: U seeds 0.00500 vs 0.00586 (±8%) | the P/U range over seed pairings 0.74-1.06 stays below 1.5 | `vera_boot.json` |
| catch metric not inflated: 5 shuffled-within-city label probes, re-set line on each state | catch 0.0-2.7% on clean AND attacked states in all 6 runs (chance ~1%) | `vera_null.py`, `vera_null_<run>_150.json` |
| ckpt scores = probe on the saved attacked states | max abs diff <= 2.1e-6 (U/P/R, all 6 runs, acts_150 readout 29) | `vera_null_*.json` consistency_* |
| specificity (cross-catch at each probe's re-set line, step 150) | after P runs U still 90.1/92.6%; after R runs U 96.6/96.3%, P 90.1/89.4%; after U runs P 36.1/29.0%, R 6.7/5.2% | `vera_null_*.json` transfer_* |
| U baseline repro (T2) | new U s0 0.00500 vs 1023 s0 at H150 0.00598 = 0.836x (< 20%); 1023 U at H300 rebuilt exactly 0.00396 | `vera_price_H150.json`, `vera_price_H300.json` |

## 3. Inputs and geometry (my code)

- `z_post.npy` = my own HF post-norm country-token states: cos >= 0.99978 on 24 random statements; cpos = my country index 24/24 (`vera_states.py/.json`).
- Clean scores of the U probe = 1023 ckpt_0 column to 8e-7; new-code ckpt_0 within 0.216 logits (p99 0.060), line -0.3443 and catch 98.15% identical.
- Mean output Fisher F̄ over the 1,400 fit-pool states (probe+attack+stop) rebuilt independently: relative Frobenius error 2.0e-7 to `Fbar.npy`,
  top eigenvalue 0.14726, eigenvalue counts above 1e-1..1e-6 of the top identical (7/26/96/398/1463/1535), 128th = 6.42e-4 of top,
  top-128 eigenspace vs `U_kstar`: min principal cosine 0.9999999996 (`vera_fbar.py`, `vera_fbar_probe_attack_stop.json`).
- Probes: `U_kstar`, `Q_kstar` orthonormal to 6e-15; P lies 100% in span(U_k), R 100% in span(Q_k); |P_U a|/|a| = 0.376 (14.2% of the
  full probe's squared length in the top 128); cos(P, U) = 0.21, cos(R, U) = 0.16 -> both refits are mostly new directions (title fixed to "restricting").
- Clean eval catch with the PREREG s5 line (2nd-highest of 200 refit true): U 98.15%, P 93.96%, R 94.63%, literal P_F w 69.8%.

## 4. Math (own derivation in `derivation.md`; numeric checks in `vera_toy_lemma.py`, `vera_fastcheck.out`, `vera_kkt.json`)

- Post display eq. 1, KL(p||softmax(Wz+Wδ)) = ½ δᵀFδ + O(|δ|³) with F = Wᵀ(diag p − ppᵀ)W: correct (log-sum-exp Hessian, logits linear in z).
- Post display eq. 2, pair price D²/(2(g_f+g_t)): correct (optimal split x = D g_f/(g_f+g_t)).
- Jensen (phase 0): 1/(aᵀF⁺a) is concave in F, so the mean-F floor overstates the per-statement floor; the design used per-statement F_i. Toy: 184x gap.
- Definition caveat (accepted by Selma, #6300): the pair price is the lowest price for THIS move model (false down + its twin up), not a bound for
  every attack on the re-set-line monitor (raising only the top-2 of the 200 line-setting true statements would hide everything at ~0 eval KL).
- Power gate, my own numbers before any P/R step (`vera_gate_np.py`, n=7 pairs, full vocab, fast Fisher checked vs fp64 to 2e-5):
  second-order false-only P/U 3.68x (paired median 3.16x), pair P/U 2.14x (paired 1.32x), R/U 0.09x. Measured: P/U 0.89x, R/U 0.50x.
  The predicted ordering R < U < P held for the floor; the measured price followed it only for R and only weakly.
- Iris's certified sphere minima checked by KKT with my code (`vera_kkt.py`): stmt 1325 U KL 5.0398e-5, P 1.9584e-5: feasible to 1e-15,
  on the sphere (|z/g|/sqrt(d) = 0.99999997 / 0.9999972), KKT residual 6e-8 / 7e-11, rho >= 0 -> certified sphere minima.
  **R (stmt 1325): the optimum is INSIDE the ball (radius 0.954 of sqrt(d)), so 4.89e-6 is only a lower bound on R's sphere minimum, not the
  minimum.** Any text that calls the R sphere number exact must say "lower bound".
- My own sphere searches are 1.65x above the certified minimum on 1325 U (8.32e-5 vs 5.04e-5): found moves are upper bounds; mine are not used.

## 5. Novelty
See `novelty.md`: floor = FishBack (2605.17231) read from the attacker's side; attack shape = Neural Chameleons / RL-Obfuscation /
Obfuscated Activations; readout-through-outputs spirit = Zhong 2026 (LessWrong). No prior work found that refits a probe inside the
output-Fisher subspace and measures an output-matched fine-tune's price against the floor ratio.

## 6. Deviations I pass with caveat (disclosed in DEVIATIONS.md)
D8 horizon 150 (convergence not reached: U KL still falling at 150; disclosed); D9/D10 5-step check failed -> U is new-code n=2;
D11 5th run start 18:30; 3 seeds planned, 2 per arm ran. PREREG s8 says "UNSETTLED if < 3 seeds per arm", so by the letter the
prediction outcome is UNSETTLED with n=2; item 3 fails by ~3x on both seeds (P all-token KL 1.07x U, needs >= 3x) and no third seed
could rescue it, so "failed" is the honest reading. The post must state that n=2 per arm and that s8's 3-seed clause was not met.
Compute: one fine-tune at a time, peak 5029 MiB under my 5100 MiB gate.
