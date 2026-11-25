role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off — llc-bias-law

**SIGN-OFF: UNCONFIRMED** (the frozen status reproduces; the answer is "not settled").

All numbers below come from my own code (this folder), written from PREREG.md 2(4)-(6) and run on the RAW data:
per-chain block means `data/cells_{S,P}/*_blocks.npz`, Gibbs rows `data/gibbs_S/*.json`, `predictions.json`.
I did not import or read the room's `score.py`, `fit.py` or `run.py` for the rebuild.

## 1. Rebuild of the frozen status (vera_score.py, vera_blind_score.py)

| quantity | room (results.json) | mine | file:line |
|---|---|---|---|
| rows rebuilt from block means | - | 1,134 arm S (+540 arm P); per-chain e max dev 2.3e-7 | vera_score.py `load` |
| halves rule (2 pts, every seed) | 29 not settled | 29; 0 flips vs `halves_ok` | vera_score.py `load` |
| train fit a, b (72 cells, q >= 0.1) | 0.0303, 0.546, rms 2.21 | 0.0303, 0.546, rms 2.21 | vera_trainfit.py |
| e_L for all 756 predicted cells | predictions.json | max dev 4e-14 | vera_predcheck.py |
| share(miss <= 5), 180 primary | 0.822 | 148/180 = 0.822 | vera_blind_score.py |
| median miss L / N1 / N2 / N3 / N4 | 1.90 / 7.40 / 2.21 / 4.63 / 2.01 | 1.90 / 7.40 / 2.21 / 4.63 / 2.01 | vera_postcheck.py |
| spread of e_meas | 49 pts | 49.0 pts | vera_blind_score.py |
| seed-level share | 0.819 [0.811, 0.826] | 0.817 / 0.822 / 0.817, t-CI [0.811, 0.826] | vera_score.py |
| status | UNCONFIRMED | UNCONFIRMED (0.80 bar met; skill bar fails: needs L <= 1.0) | vera_score.py `score` |

## 2. Independent sampler (vera_blind_sgld.py, vera_blind_compare.py, vera_train_gibbs.py, vera_s2b.py)

Own teacher and data draws, own minibatch SGLD (4 chains) and own block Gibbs, frozen T map; run after the
predictions push (14:31:14Z). 15 blind cells: my SGLD minus the room's e within 1.2 pts (median +0.6); my Gibbs
minus the room's within 0.3 pts. Train Gibbs at nbeta 1e3: within 0.3 pts. Parent spring (gamma 10, nbeta 26.38),
noise-free: my Gibbs -30.1/-30.2% (r1), -33.0/-33.2% (r8); the post says 30% to 33%.

## 3. Null tests

- Shuffled q (vera_robust.py `shuffle_null`): q permuted across the 72 train cells, curve refit, scored on the
  blind core, 300 perms: share 0.59 mean, 95th pct 0.71, vs 0.822 real (p = 0.003). The q curve carries signal.
- Lookup null N4 (same setting on a training shape): median miss 2.01 vs L 1.90. On settled cells N4 is better
  (1.48 vs 1.73). At a fixed setting the 5 blind shapes differ by a median 4.5 pts (8.1 at q 0.1, 2.2 at q 100), and
  the setting alone explains 93.6% of the variance of the measured error (vera_shape_spread.py). The blind core
  therefore barely tests transfer across shapes.
- The status edge is thin: with the frozen reading (a failed cell is a miss of 100 for every predictor; score.py at
  3514192de, before data) L 1.90 < N4 2.01, so not REJECTED. Seed 1 alone: L 1.95 >= N4 1.90. Cell bootstrap
  P(L >= best null) = 0.37. With the 100 applied to the law only: N4 1.64 and N2 1.86 < L 1.90, REJECTED
  (vera_robust.py).
- Settle-rule noise (vera_halves_noise.py): from each cell's own 4-chain spread, pure Monte-Carlo noise predicts
  29.4 flagged cells in arm S; observed 29 (drift z mean +0.13). The 2-point rule with 4 walks flags ~16% of
  settings by chance, and those flags decide the 0.80 bar. Arm P: 29 observed vs 18.9 from noise.

## 4. Derivations (vera_equations.py, vera_aoyagi.py, vera_split.py)

E1 tethered ULA variance 1/(a(1 - eps a/4)) and the regular term 1/2 * 1/(1+g) / (1 - c/4): sympy exact. E2 null-block
rescaling w = nbeta^(-1/4) u: depends only on gamma/sqrt(nbeta) = (2q)^(-1/2): sympy exact. E3 Gaussian limit
nbeta E[K0] = q M'N'H': Monte Carlo within 0.1%. E4 X scales as s^4; q invariant once k4 carries s^4. Aoyagi values
236 (m=2) and 312 rebuilt; split identity 0 mismatches on 78,744 shapes (M, N <= 24); it is Lehalleur & Rimanyi
2411.19920 Lemmas 4.5-4.6 (prior work, credited in the post).

## 5. Definitions not stretched

q = nbeta/(2 gamma^2) as frozen; h_i population {1, 2} as frozen (2(1)); the false '1/sqrt(n)' sentence is corrected
in DEVIATIONS.md. The secondary L_z (median 1.4) is labelled "added after a pilot" and carries no status weight.

## 6. Caveats I require in the post

1. The law does not beat the lookup (1.90 vs 2.01; settled-only 1.73 vs 1.48).
2. The status margin is 0.11 pt and depends on how failed cells are scored (law-only reading gives REJECTED).
3. The 29 unsettled flags match chain noise (29 vs 29.4 expected).
4. Pre-freeze pilot (mine, vera_power_v2.py, 06:03 PST) touched the seven "seen" shapes, not the blind core.

## 7. Novelty

See novelty.md. The split is Lehalleur & Rimanyi's; the tether-changes-the-target idea is in Cai 2606.22389 and
Lau et al. 2308.12108 (qualitative). New: the q-only dependence of the null block and a fitted curve scored on
unseen shapes.
