# T3 — theory note and toy rerun with the real settings (Iris, 2026-11-10, before any Qwen RL run)

Files: `toy_real.py` (sweep, 61 s, numpy, CPU), `toy_real.json` (all runs), `fit_toy.py` -> `fit_toy.json` (fits).
Rebuild: `uvx --with numpy python toy_real.py && uvx --with numpy python fit_toy.py && uvx --with numpy python toy_sens.py` (toy_sens.json: spread and KL-weight checks, 18 s).

## Settings copied from the organism (stack-conditional rl.py, origin/swarm/oracle-1030-stack-conditional)
P = 8 prompts per update, cue flag ~ Bernoulli(1/2) per prompt, G = 8 actions per prompt; reward -1 for X on a
penalised (= cue-on) episode, 0 otherwise; advantage (r - mean) / (unbiased std + 1e-6); loss -sum adv log pi / (P G)
+ 1.0 * exact KL(pi || pi_ref) / P; global grad-norm clip 1.0; AdamW lr 1e-4, wd 0; LoRA r 16, lora_alpha 32,
B = 0, A ~ U(+-1/sqrt d). Switch: (gap - gap0) >= 0.10 at two evals in a row (every 4 updates), T = the first.

Toy simplification (mine): binary action, logit z = c_p + 2 w^T B A x_p, d = 64, x_p = mu + sigma xi_p (sigma 0.7),
base P(X) 0.39 on average; cue-on input x_p + alpha sqrt(d) u. alpha = cue size relative to the input RMS norm.
Budget 40,000 updates. 11 alphas 0.01-3 (log-spaced) x 3 seeds x {AdamW, SGD}. SGD lr = 0.0032 from a rule fixed in
code before the SGD runs: step-0 update norm at alpha_top equal to AdamW's step-0 update norm (`sgd_lr()`).

## Results (fit_toy.json)

| quantity | AdamW (real optimiser) | SGD (rule lr) |
|---|---|---|
| OLS slope of log T on log alpha, 18 uncensored runs, alpha 0.17-3 | **-1.40** (95% CI -1.50 to -1.29) | **-1.86** (95% CI -2.01 to -1.71) |
| slope, bottom half of alphas / top half | -1.59 / -1.08 | -2.02 / -1.50 |
| local slopes, small -> large alpha | -1.57, -1.53, -1.53, -1.03, -1.22 | -1.88, -1.78, -2.10, -1.99, -1.42 |
| alphas censored at 40,000 | 0.01-0.098 (5 of 11), all seeds | the same 5 |
| late gap of censored runs (flat from ~10k to 40k) | 0.001, 0.003, 0.011, 0.032, 0.083 | 0.001, 0.003, 0.011, 0.033, 0.083 |
| slope of log(late gap) on log alpha | 1.78 | 1.78 |

Controls: alpha = 0: no switch in 40,000. Shuffled penalty (independent of the cue) at alpha = 3: seed 0 crossed
0.10 at update 36 by noise; seeds 1-2 and alpha 1 / 0.3 stayed within +-0.025 over 8,192 updates.

## What the toy says (each line is checked by a number above)

1. **In this binary-output toy, SGD gives about -2 and AdamW about -1.4; other toys differ, so the Qwen SGD arm
   decides.** Here the per-update change in the gap under SGD is prop. to lr alpha^2 (linear B phase), and the slope
   does not move with the SGD learning rate (Kasper's rebuild: lr 0.0032 -> -1.97, lr 0.032 -> -1.95). AdamW divides
   each weight's step by its recent gradient size, which the alpha-free "flatter less everywhere" push dominates;
   this toy gives -1.5 at small alpha and -1.1 at large alpha. This is NOT a general "optimiser sets the exponent"
   result: Bruno's real-settings toy with no spread along the cue (projects/cue-size-law data/toy_real_nospread.json)
   gives about -1.1 for SGD and AdamW alike, and there the -1 comes from LoRA A learning (two factors; freezing A
   makes alpha 0.5 take 40,064 / 9,312 updates for +u / -u instead of 1,344, Kasper #19376). [GUESS, untested] the
   toys differ in the prompt spread along the cue (mine 0.7 isotropic; Bruno's 0 along u) and in the output (binary
   vs 16 tokens with a random readout); either may cause the gap. P2 stays MY toy's number.
2. **A fixed-size switch has an edge, and the KL pull sets it, not the optimiser.** Below alpha_c ~ 0.11 the gap
   rises and then STOPS at a level prop. to alpha^1.8 (about alpha^2), the same for SGD and AdamW. The pull to the
   start policy acts on cue-off episodes. The model reads the cue with a linear map that also picks up prompt-to-
   prompt variation along the same direction, so the pull charges for every bit of switch: a ridge-regression
   balance. Saved checks (toy_sens.py -> toy_sens.json, 2 seeds each, budget 20,480, late gap = mean over the last
   half): at alpha 0.055, sigma 0.35 -> crosses 0.10 (T 13,808 / 9,984 for s0 / s1), sigma 0.7 -> 0.032/0.033,
   sigma 1.4 -> 0.0081/0.0084; KL weight 0.5 -> 0.041/0.043, 1 -> 0.032/0.033, 2 -> 0.018/0.018. At alpha 0.098:
   sigma 0.35 and beta 0.5 cross (T 1,484-1,692 and 5,660-5,688), sigma 1.4 -> 0.026/0.027, beta 2 -> 0.046/0.046.
   Log-log slopes: late gap vs sigma -1.97 (alpha 0.055) and -1.64 (alpha 0.098); vs beta -0.62 and -0.83. So, in
   this toy, late gap ~ C (alpha / sigma)^(1.6-2) / beta^(0.6-0.8), and a 10-point bar gives an edge alpha_c that
   grows with sigma and with beta. T diverges as alpha -> alpha_c from above.
3. **A slope near the edge is not a clean law.** Close to alpha_c the time to a 10-point gap grows faster than any
   power, so a fit that includes alphas near alpha_c steepens; a fit far above it flattens toward -1 under AdamW.

## Frozen predictions for the Qwen3-0.6B run (posted before T0)

- (P1) AdamW grid, uncensored alphas well above the edge: slope between -1.0 and -1.6 (toy -1.40); bottom-half slope
  steeper than top-half slope. The prereg band [-2.5, -1.5] is then missed or only grazed: I put P(prereg holds) at
  ~0.3, below the scout's 0.45.
- (P2) SGD arm (lr by the same rule): slope -1.7 to -2.1 (toy -1.86).
- (P3) Censored alphas: the gap curve goes FLAT over the last half of the budget at a level that falls ~alpha^2 (log-log
  slope 1.4-2.2) and does not depend on the optimiser. That is the KL cap, not slow learning. Read: "the cue is
  learned, but the pull caps the switch below 10 points"; this is the prereg's "threshold" outcome with its cause.
- (P4) Regime map from toy to Qwen: R(alpha) = ||E g_B(alpha) - E g_B(0)|| / ||E g_B(0)|| at step 0 (toy: R = 1.26
  alpha; bend at R ~ 1.2-2; edge at R ~ 0.14). Bruno logs the same R per alpha in T2; the Qwen bend is predicted at
  the alpha where R reaches 1.2-2. The edge depends on sigma and beta, which the toy cannot transfer, so it is
  measured from P3 plateaus, not predicted in alpha units.
- (C1) Do not put alpha_top above about 1x the row norm without 2 shuffled-penalty seeds there: a huge cue makes the
  cue-on inputs a separate cluster, and noise can push the gap over 10 points (toy alpha 3, seed 0, update 36).

## Prior work this rests on (sources in ../research.md)
Saxe, McClelland, Ganguli 2014 (arXiv 1312.6120): plateau-then-jump learning of a mode in deep linear nets, time
~1/strength. Balles and Hennig 2018 (arXiv 1705.07774): Adam = sign of the gradient times a variance factor, so the
step depends on the gradient's signal-to-noise. Roberts and Tedrake 2008 (NeurIPS): signal-to-noise analysis of policy
gradients. GRPO advantages: Shao et al. 2024 (arXiv 2402.03300). LoRA: Hu et al. 2021 (arXiv 2106.09685).

## Addendum 06:10 PST (pre-data, after Bruno T2 scan #19381, room #19385)
Qwen scan: per-weight cue SNR median 0.098 at alpha_top 1/32 (~0.8 only at alpha 1); R = 0.084, 0.039, 0.019,
0.0096, 0.0047, 0.0027 for k = 0..5. Toy bend at R 1.2-2 -> alpha* ~ 1, outside the sweep: one regime, no Adam bend
expected inside k = 0..5. Toy edge at R ~0.14 is NOT transferred (it depends on the spread along the cue; Qwen rows
0-2 have zero prompt spread, and the real cue at x 11.69 switched in stack-conditional). Reading frozen: a censored
alpha with a flat last-half gap = the cap; late gaps across censored alphas fall ~alpha^(1.6-2); a still-rising gap =
too short a run.

## Addendum 2 (room #19414, after sweep seed 0 T = 32..704, slope -0.90; BEFORE cell 4c (-u) runs)
Cross-term mechanism. Adam step on weight i: lr E[g_i]/rms(g_i), E[g_i] = common_i + alpha c_i. Gap ~ linear in the
cue: dgap/dtheta_i = alpha d_i. Per step: dgap = lr alpha SUM d_i common_i/rms_i + lr alpha^2 SUM d_i c_i/rms_i. The
first (cross) term = the global "flatter less" update read through the cue; it gives T ~ 1/alpha and flips sign with
u -> -u; the second gives T ~ 1/alpha^2 and does not. Frozen: at k = 2, T(-u) >= 2 T(+u) or -u censored. If
T(-u)/T(+u) in [0.5, 2], the cross term is not the mechanism.

## Addendum 3 (exploratory; arrival size d; room #19447, #19480, #19487)
- Frozen #19447 (07:15 PST, before cells 13-14): row-0-only T <= 2x T(k=1) and rows-1-2-only T >= 192 at x 16.556.
  Result: 32 and 1,472 (Bruno #19467). Both held. Basis: gain_check.json (keys change mostly on row 0 for the
  planted cue: dK row 0 3.74 vs rows 1-2 1.47/1.78; real cue dK row 0 0.149).
- transport.py: exact block-26 attention at the answer token (row 33), 64 dev prompts. Attention share: row 0 0.668,
  rows 1-2 0.0023. Arrival size d = ||o_on - o_off|| (median): balanced k0..k5 16.43/7.63/3.73/1.84/0.90/0.49;
  row0-only 14.29; rows12-only 0.048; real 0.149. Post-hoc: the balanced law in d predicts ~2,060 for the real
  cue; observed 1,216 (0.59x). Rows12-only is 3.9x faster than the balanced law in d (training can open attention
  to rows 1-2).
- Frozen #19480 (07:41 PST, before the direction cells): rows-1-2-only u seeds 1112-1116 at x 11.695 have d
  0.025/0.040/0.025/0.030/0.026 (transport_dirs.json). Predicted (a) all T > 1,216; (b) median in [1,700, 3,500];
  (c) 1113 fastest; points T = 1,472 (0.048/d)^0.9. Results (n = 3): 2,880 / 1,728 / 2,240 for 1112/1113/1114;
  (a)-(c) held; points 1.08x / 1.00x / 0.84x.
