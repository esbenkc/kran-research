# Vera (skeptic) sign-off — theory-1023-norm-rescale-free / norm-rescale-free

Verdict: **REJECTED (no to the title) — CONFIRMED by my rebuild** under PREREG s6 (+ DEVIATIONS N1: 596 eval twin pairs).
Prediction outcome: **FAILED** (clauses 1, 2, 4 fail; clause 3 holds and is vacuous).

All numbers below come from my own code on build's RAW saved files (`data/runs/<run>/acts_<step>.npy` fp16 country-token
states, `ckpt_<step>.npz` per-position KL / koff / cpos), the frozen 1021 probe weights `panel.npz` and the frozen split.
I did not read build's scripts or results.json. Build's saved scores were used only as a cross-check (my tok28/tok29 scores
match them to <= 1.3e-6 at every checkpoint).

## Title metric (PREREG s6): norm-restore share R, arm A (pre-norm tok28), lambda 1e4, step 300

| seed | g_clean | gap drop | precondition | **R** | city-bootstrap 95% (1000) | max R over steps 50..300 |
|---|---|---|---|---|---|---|
| 0 | 9.085 | 99.8% | met | **0.0101** | 0.0094-0.0106 | 0.0107 |
| 1 | 9.085 | 98.5% | met | **0.0143** | 0.0137-0.0149 | 0.0143 |
| 2 | 9.085 | 98.6% | met | **0.0085** | 0.0079-0.0090 | 0.0094 |

All 3 seeds: precondition met and R < 0.5 -> **NO** (`vera_check.py`, `vera_check_r28__lam10000__seed{0,1,2}.json`).
Build #5486/#5522/#5527: 0.010 / 0.014 / 0.0085. Same.

## Independent forward pass (my model load, my hook; `vera_fwd.py`, CPU bf16, build's adapter_300, 40 random eval pairs = 80 statements)

| | seed 0 | seed 2 |
|---|---|---|
| my readout 28 (hook on layers[27], pre-norm) vs build acts_300 | cos >= 0.99971 | cos >= 0.99973 |
| my readout 29 (HF hidden_states[28], post-norm) vs build | cos >= 0.99993 | cos >= 0.99992 |
| my clean readout 28 vs build acts_0 | cos >= 0.99979 | (same) |
| per-position KL, my mean vs build kl rows (same 80 statements) | 0.00842 vs 0.00843 (corr 0.9994) | 0.003249 vs 0.003250 (corr 0.86) |
| R on those 40 pairs: from MY states vs from build's states | 0.0098 vs 0.0094 | 0.0080 vs 0.0078 |
| tokens per statement match koff | 80/80 | 80/80 |

Step-0 states also match the 1021 clean acts (cos >= 0.99985, all 2992 statements).

## Mechanism and catch rows (my code)

| run (step 300) | R_common | R Shapley (length vs direction) | R forward (length change only, on clean) | energy share E | norm ratio false/true | tok28 catch re-set / clean line | tok29 catch re-set | eval KL (country) |
|---|---|---|---|---|---|---|---|---|
| A s0 | 0.0000 | 0.031 | 0.051 | 37.9% | 0.951 / 0.970 | 1.85% / 51.7% | 1.85% | 0.00448 (0.00334) |
| A s1 | 0.0002 | 0.027 | 0.039 | 32.2% | 0.963 / 0.985 | 1.34% / 55.7% | 1.85% | 0.00579 (0.00424) |
| A s2 | 0.0004 | 0.027 | 0.046 | 31.5% | 0.956 / 0.971 | 2.01% / 50.8% | 2.68% | 0.00486 (0.00393) |
| B s0 | -0.0001 | -0.028 | -0.041 | 20.2% | 1.039 / 1.014 | 2.01% | **0.84%** (own) | 0.00373 (0.00374) |
| B s1 | -0.0003 | -0.031 | -0.044 | 23.5% | 1.042 / 1.014 | 1.85% | **1.68%** | 0.00413 (0.00366) |
| B s2 | -0.0003 | -0.027 | -0.040 | 17.9% | 1.039 / 1.019 | 2.18% | **2.18%** | 0.00507 (0.00449) |

(B rows: R uses tok28, the probe arm B did not attack: the null.)
Restored states (every attacked length set back) are still hidden: tok28 re-set-line catch after restore = 1.85% / 1.34% / 2.35%
(attacked: 1.85% / 1.34% / 2.01%). Pooled arm A: tok28 1.73% (re-set), 52.7% (clean line), tok29 2.13%, eval KL 0.00504.
Pooled arm B: tok29 1.57%, tok28 2.01%, eval KL 0.00431.

**Order of the decomposition does not change the answer.** The frozen R restores at the END state; Shapley and forward
splits give 2.7-5.1%. All << 0.5.

## Prediction clauses (PREREG s7), my numbers

1. pooled A catch < 20% AND KL <= 0.002: catch 1.73% but KL 0.00504 -> **FAIL**. (The one-Adam-step KL is already 0.0021, smoke log.)
2. pooled A energy share > 70%: **34.1%** -> **FAIL**.
3. random direction < 5%: 0.063% (1000 random units, seed 1023) -> holds, vacuous by construction (1/d = 0.065%).
4. K per seed (min eval KL over ckpts 50..300 with re-set catch < 20%): A 0.00397 / 0.00435 / 0.00424, B 0.00373 / 0.00388 / 0.00491;
   ratio of medians B/A = 0.00388/0.00424 = **0.91** (needs >= 2) -> **FAIL**.

## Nulls and calibrations

- **Planted-signal calibration** (`vera_planted.py`): take a fraction f of A s0's real edit and add a falsity-dependent rescale
  that closes the rest of the gap. R recovers the planted rescale share: planted 1.00 / 0.75 / 0.50 / 0.25 / 0.10 ->
  R 1.000 / 0.761 / 0.516 / 0.265 / 0.113. So R sees a rescale share of 10% when one is there; the fine-tune's is ~1%.
- **Arm B null** (radial moves useless against tok29): R -0.016 / -0.019 / -0.013. Arm B LENGTHENS states (1.04x) yet shows
  18-24% radial energy: the energy share is not evidence of the free route.
- **Earlier attacks not built to find the rescale** (`vera_oldstates.py`): 1022 panel R -0.006..+0.000 (Iris #5439 same);
  1021 layer-18 attacks R 0.03-0.05 (their precondition at readout 28 is not met, drops 9-31%).
- **Different seeds**: 3 of 3 agree (R spread 0.0085-0.0143).

## Math re-derived (`derivation.md`, `vera_lemma.py` on the real final norm + tied unembedding, fp64)

- Rescale c > 0: max KL 1.4e-13 (c in {0.3, 0.55, 1.8, 3.0}); c = -1 costs 32 nats (a sign flip is NOT free).
- Euler h.grad KL = 0 at perturbed points: |cos| <= 5.7e-10. Hessian at clean: radial curvature 5e-20 vs tangential 5e-5..1.6e-4.
- Norm restore: KL 4.5e-14, post-norm state moves 1.2e-8 relative.
- Step-0 activation-space prediction: R0 = mean cos^2(w/sd, h_hat) over eval twins = 0.27% (false median 0.51%, true 0.004%);
  finite steps 0.24-0.27%. Iris's gap-weighted 0.28% agrees. The LoRA's measured R is 3-5x higher (weight space tilts toward
  the shrink), still 35-60x below 0.5.
- Exact identity used by the 'radial share of the score change' row: s(h') - s(h) = (c-1) a + v.t; that row (9-11%) counts the
  drop in projection caused by TURNING (c = cos(theta) |h'|/|h|) as radial, so it overstates the length's role; the restore
  test does not.

## Checks on claims made in the room

- lambda_sim ~ lambda/21 (#5503/#5506): arithmetic holds for country-token-only KL (8 statements x 10.65 tokens = 85 positions,
  /2), but only 3.5% of the LoRA's summed per-position KL on attack statements sits at country tokens (A s0 step 300). The x-position
  is a loss-weight match only; say so wherever it is used.

## Caveats (disclosed, do not change the headline)

- One KL weight (1e4) decides the status; secondary runs (3e8, 1e5, lam 0) are n = 1 (rebuilt below).
- Forward-pass check covers 2 of 3 arm-A adapters at step 300 on 80 statements (CPU bf16 vs build's MPS).

## Secondary runs (n = 1 each, never status; rebuilt 18:15 CEST with the same `vera_check.py`)

| run (step 300) | gap drop | R | R_common | Shapley | forward | tok28 catch re-set / clean | tok29 re-set | restored catch | eval KL | norm ratio false/true |
|---|---|---|---|---|---|---|---|---|---|---|
| A lam 3e8 | 2.3% (precondition FAILS) | 0.0195 (meaningless) | -0.029 | 0.019 | 0.019 | 98.2% / 98.8% | 98.2% | 98.2% | 0.00327 | 1.000 / 1.000 |
| A lam 1e5 | 70.0% | 0.0206 | 0.0199 | 0.044 | 0.068 | 43.3% / 80.0% | 41.1% | 43.5% | 0.00566 | 0.953 / 0.954 |
| A lam 0 | 100% | 0.0011 | -0.0002 | 0.172 | 0.344 | 0.17% / 0.0% | 0.17% | 0.0% | 0.986 | 0.658 / 0.663 |

- N2 frozen prediction for 3e8 ('R >= 0.5 OR gap fails to halve') holds only through its second branch: the attack did not move.
- lam 1e5: 10x the weight leaves R at 2%, and that 2% is almost all a COMMON shrink (R_common 0.0199).
- lam 0: the order of the split matters here (R 0.1% vs Shapley 17% vs forward 34%): a 0.66x shrink scales every raw twin gap
  by 0.66 because b0 cancels in the pair difference. The catch test is order-free: restoring every length leaves 0.0% caught.
  So the shrink bought no hiding at the alarm line; it does not mean the shrink left the raw gap untouched.
