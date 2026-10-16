# skeptic — SIGN-OFF on NULL (2026-10-15, 23:20 CPH)

Claim checked: "With SGLD settings frozen after calibration, the measured LLC of grokked quadratic
networks on modular addition is 0.65-0.78 of the closed form K_eff(3p-1)/2 (Cullen et al. 2603.01192 v3,
Remark 5.5) at p = 11-23, with log-log slope ~1; pre-registered rule -> NULL."

Fresh context: my /tmp work from the day was lost in the reboot. Everything here is rebuilt from the
raw checkpoints (data/ckpt/*.npz) and data/llc.csv + data/stationarity.csv. I did not read or import
build/plan code. Own model, loss, gradient (FD-checked, max err 1.9e-8), Jacobian, polish
(Levenberg-Marquardt instead of build's Adam), SGLD loop and chain seeds.

## 1. Derivation (derivation.md, derivation.py -> derivation.json)
- Loss convention identified from saved train_loss: residual centred over training points, idx -> (idx//p, idx%p) (identify_loss.json).
- Per-neuron count on one-hot data is 3p-2, not 3p-1: exact integer rank, p = 3,5,7,11 -> 7,13,19,31; scaling and shift tangents in the kernel.
- Generic K-neuron rank = min(K(3p-2), p(N-1)) in 9/9 exact cases. V-only rank = p*min(l,K) in 4/4.
- Shift gap K/2, relative 1/(3p-1) <= 5%. SGLD quadratic limit nb h/(2(nb h+g)) (sympy). All post arithmetic (6,732; 3,784; 1,700; 528; 3,128) checks.

## 2. Rerun from raw checkpoints (rerun.py -> rerun.jsonl), frozen setting nb=1e4, g=1, crel=0.2, 10k steps, 4 chains
25 checkpoints: all p7/p11 grok + mem, p17_K51_s0, p17_K68_s0 (single process, OMP=1, <=115 MB).
- Mine vs llc.csv lambda_hat: max |diff| 2.7%, median 0.0% (verdict.json "mine_vs_build").
- Inside-hypothesis grok cells, mine: p11_K33 s0/s1/s2 = 0.719 / 0.710 / 0.724; p17_K51_s0 0.743; p17_K68_s0 0.655 of K_eff(3p-1)/2.
- Memorising, first layer frozen: 0.94-1.01 of p*min(l,K)/2 (12 ckpts). p11_K88 mem reads 0.94 (resolution, lambda_quad 0.93).
- Different polish, same reading: my LM polish moves weights up to 5% and lowers the numerical rank (p11_K33: 960 vs 1023) and lambda_quad (p17_K68: 977 vs 1088), but lambda_hat moves < 3%. The post's [^sloppy] now says the flat-direction share depends on polish and the reading does not. Correct.

## 3. Verdict recompute (verdict.py -> verdict.json), my own inside-hypothesis test and cell grouping
- Main (p11-23, inside, settled at 40k, p31 frac0.5 out): 5 cells / 12 networks, median 0.721, range 0.655-0.782, 0/5 cells in the 15% band, slope 0.985 -> NULL. Every seed < 0.79. Max ratio vs corrected K_eff(3p-2)/2: 0.794 -> NULL also against the corrected count.
- Robustness: with p31 frac0.5 added: 1/7 in band, slope 1.078 -> NULL. With p31 frac0.7 (2 seeds so far, 0.87/0.85): 1/6 in band -> NULL. The verdict does not depend on p31.

## 4. Null tests (null.py -> null.jsonl)
- N1 random-weight planted net (same p, K, train set, weight scale; targets = its own outputs, so known lambda = rank/2):
  p17_K51: reads 0.927 of rank/2, 0.04% of directions below resolution (post's 0.04% reproduced). Grokked p17_K51 at the same setting: 0.797 of my rank/2, 24.9% below resolution (post's 24% reproduced). So at p17 the shortfall is mostly specific to the grokked point.
  p11_K33 (near the p(N-1) limit: 1023 vs 1056): planted net reads only 0.838 of rank/2 (lambda_quad 0.909). So near the limit even a random net reads low; grokked p11_K33 reads 0.795. At p11 the grokked-specific part is small.
- N3 chain-seed null: fresh chain seeds at the grokked ckpt: p11_K33_s0 381.6 vs 379.6; p17_K51_s0 947.3 vs 947.4. Chain luck is not the story.
- Training-seed spread inside cells: <= 0.6% (p11_K33 0.718-0.723; p23_K92 0.675-0.676).
- N2 (student trained from scratch on a random teacher) FAILED to reach zero loss (L = 7e-6 at p11, 3.6e-3 at p17), so it is not a valid known-lambda point. Excluded; reported for completeness.

## 5. Novelty (novelty.md)
No prior calibrated test of Cullen et al.'s absolute value. v3 already reads absolute LLCs "up to an estimator-dependent scale", so the new content is the size (~0.72) and constancy over p = 11-23 of that scale. Panickssery & Vaintrob 2023 (different model) saw SGLD ~2x ABOVE a heuristic count. No source mentions the one-hot shift symmetry.

## Definitions stretched?
No for the verdict: K_eff rule fixed before LLCs, inside-hypothesis rule pre-registered, settled rule applied, p31 exclusion justified by stationarity + frac confound and verdict-neutral.

## Conditions on the post (wording, not the verdict)
1. TLDR/excerpt "within 7% on networks where we know the answer": not general. My planted random-weight net at p11_K33 (a network where we know the answer) reads 0.84. Write "within 7% on the networks we checked it on" or name the set (8 calibration nets + frozen-first-layer memorising nets).
2. If p31 frac0.7 rows enter the figure (~0.86), "at every size" / "parallel line" / "about 70% from p=11 to 23" must say the share is ~0.85-0.87 at p=31 (skeptic2 #781 wording OK).
3. Caption [N] filled; figure title/summary.json must match the p11-23 verdict numbers above.

SIGN-OFF: NULL holds. Numbers: median 0.72 (0.655-0.782), 0/5 cells in band, slope 0.985; my independent rerun within 2.7% of llc.csv on 25 checkpoints.

## Follow-up 2026-10-16: 'Sampler and calibration (for SLT readers)' fold (G. Wang feedback)
Files: sampler_check.py -> sampler_check.json (A-E), sampler_check_F.json, sampler_check_F_gamma.txt,
long_gamma.py -> long_gamma.json, nbeta_table_check.txt, summary sampler_check.md. Own code throughout.
- Calibration nets rebuilt from the stated recipe: rank/truth/h_min/eps identical 8/8; my SGLD est/true -0.9..+1.7%.
- eps rule holds on 53/53 llc.csv rows. Grid: frozen setting 4th of 16 by worst error (3.87%), grid flat.
- Finite-chain quadratic model: 10k chain reaches 95-99% of its equilibrium reading at gamma 1.
- Sensitivity (3 nets, my code): gamma 0.1/1/10/100 -> -0..+1% / 0 / -6..-8% / -32..-42%; crel 0.1-0.4 <= 1.5%;
  nbeta 1e3/1e5 as in skeptic3's sweep. 100k-step chains: gamma 0.1 vs 1 = +1.7% (p11_K33_s0), +3.8% (p17_K51_s0).
  Max ratio over all my runs 0.776: no tested setting moves these cells into the 15% band.
- Verdict NULL unchanged. Fold numbers checked line by line (room #1245); sign-off on the fold conditional on those fixes.
