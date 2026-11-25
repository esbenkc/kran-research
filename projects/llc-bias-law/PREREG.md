# Pre-registration — llc-bias-law

Room: `theory-1125-llc-bias-law`. Night: 2026-11-25. Slug: `llc-bias-law`. Track: theory, agenda T11.
Branch: `swarm/theory-1125-llc-bias-law`. Written by Iris (research). The frozen set is Felix #30717 exactly
(confirmed by Felix #30725 and Tove FINAL #30719): Felix's v2 text (`swarm-theory-1125-llc-bias-law/PREREG_v2_draft.md`,
06:07 PST) with the blind core (Tove #30704, Kasper #30703, #30708, Vera #30700, #30710) and Bruno's T rule (#30716).

The push time of the commit that adds this file (recorded by `oracle.py prereg` in the header) is the freeze time.
No scored cell of this project runs before it, and nothing at all runs on a BLIND shape (section 2(5)) before
`predictions.json` is pushed. The pre-freeze pilots are listed in section 4 and in `DEVIATIONS.md`.

Room record: Tove #30663, #30675, #30685, #30687, #30692, #30697, #30701, #30704, #30712, #30714, #30718, #30719; Felix #30665,
#30673, #30679, #30686, #30698, #30705 (withdrawn), #30709, #30715, #30717, #30721 (withdrawn), #30725; Iris #30672, #30688, #30702, #30707; Kasper #30668, #30677, #30680, #30683, #30691,
#30693, #30694, #30699, #30703, #30708; Vera #30669, #30674, #30678, #30681, #30682, #30696, #30700, #30710;
Bruno #30666, #30670, #30676, #30684, #30706, #30716; Selma #30667, #30671.

## 0. Prediction

### Scout version (v1), verbatim from the goal (fixed 2026-11-25 14:53 CET; kept on record, superseded by v2)

> Prediction: On held-out ranks, the collapse predicts the relative error within 5 percentage points for at least
> 80% of settings. The common default settings miss by more than 10% at the two higher ranks.
> P(prediction holds): 0.45

Tove vetoed the scout form (#30685): its number X = eps*nbeta/gamma scales as s^4 under the reparametrisation
w -> s*w (Iris `research/invariance.py`), and the parent's frozen grid already spreads 60-110 points at equal X
(Kasper #30668; parent `llc-early-alarm-8b/data/gate/grid.json`). X stays in this run as null N3 and secondary S1.

### PRE-REGISTERED PREDICTION v2 (Felix + Iris; text frozen by this push; Tove's APPROVE in the room)

The split law L (section 2(4)), with its 2 constants fitted on the two training shapes only, predicts the measured
relative error of the SGLD learning-coefficient estimate within 5 percentage points for at least 80% of the primary
cells on five BLIND shapes (unseen ranks and widths), and its median miss is at most half that of every null.

P(prediction holds = CONFIRMED) = 0.50 (Felix #30717, Tove #30719: the pilot's 0.917 on seen shapes with a perfect
sampler (Vera #30710) argues up; untested minibatch noise, the c = 0.1 step bias and a new Aoyagi branch
((40,24,12) r4) argue down).
If it holds: once the regular part is taken out in closed form, the localisation bias of the standard SGLD estimator
on this known-answer model follows one curve in q = nbeta*k4/gamma^2; a gamma rule follows (section 2(7)), and the
T11 rerun can set gamma by formula.
If not: the bias depends on block shape (or on more than one number); every LLC claim needs its own known-answer or
exact-Gibbs gate. The secondary law L_z (2(7)) then says whether the derived shape-scaled variable repairs it.

## 1. Facts known before this freeze

- Aoyagi & Watanabe (2005) closed form for reduced-rank regression (RRR): `research/rrr_lambda.py` (copied from the
  parent `llc-early-alarm-8b`; reproduces Drton & Plummer arXiv 1309.0911 Table 1). lambda(32,32,16,1) = 236,
  lambda(32,32,16,8) = 312.
- Split identity of the closed form (Felix #30679; Iris 7,812 cases M,N,H <= 12; Vera 78,744 cases M,N <= 24 with
  independent code; 0 mismatches): lambda(M,N,H,r) = r(M+N-r)/2 + lambda0, lambda0 = lambda(M-r, N-r, H-r, 0).
  An identity of the closed form checked on a box, not a proof.
- The Hessian of K at the special w* has exactly r(M+N-r) positive eigenvalues (Kasper #30677: 63 at r=1, 448 at
  r=8; Bruno #30684: 124 at r=2).
- Truth holds only at a special point (Hitchcock & Hoogland arXiv 2507.21449 App. B, from Lehalleur & Rimanyi 2024):
  the local learning coefficient equals Aoyagi's global value at the "special" parameters (each factor of rank r,
  every other row and column exactly 0); at a generic point of the true set it is larger (their Thm B.5).
- Exact rescaling (Kasper #30691): in the null block K0 = 1/2 ||B A||^2, substitute w = nbeta^(-1/4) u; then
  nbeta*K0 + gamma/2 ||w||^2 = 1/2 ||B_u A_u||^2 + (gamma/sqrt(nbeta))/2 ||u||^2. So the equilibrium null-block
  estimate depends on (nbeta, gamma) only through q and the block shape.
- Strong-spring limit (Gaussian prior dominates): A, B ~ N(0, 1/gamma) entrywise, so
  E[nbeta * 1/2 ||B A||^2] = nbeta M'N'H' / (2 gamma^2) = q M'N'H' (M' = M-r, N' = N-r, H' = H-r). Hence
  psi -> z := q M'N'H' / lambda0 as z -> 0 (Kasper #30691, #30699). z is the variable of the secondary law L_z.
- Regular term: the exact stationary variance of the unadjusted Langevin step with a Gaussian tether on a quadratic,
  Var = 1/(a(1 - eps*a/4)), a = nbeta*h + gamma (prior: Cai arXiv 2606.22389 App. B.1; sympy check
  `research/invariance.py`).
- Pilots: at equilibrium the error is mostly target bias from localisation (exact Gibbs vs SGLD, W32 r1,
  gamma/nbeta = 0.1, nbeta 100/1000/10000: -16.3/-38.0/-69.5% vs -15.9/-37.7/-69.2%; Vera #30678, #30682). On the
  training shapes the SGLD-Gibbs gap closes when eps*T >= ~0.1 (Bruno #30706: c 0.01, nbeta 1e4, q 100, r1: 50k
  steps -4.1, 100k -2.3, 200k -0.9 points).

<!-- FROZEN RULE -->
## 2. Frozen rule

### Question (yes/no, verbatim in the post)
Once Aoyagi's formula splits reduced-rank regression into regular and singular parts, does one curve in
nbeta/gamma^2 predict the SGLD learning-coefficient error at held-out ranks and widths within 5 percentage points?

Scope: equilibrium error only (chains run until they settle); scored arm S = minibatch SGLD on a noise-free teacher.
The law licenses no chain-length budget.

### (1) Model, losses, truth, start point
- Model y = B A x, A: H x M, B: N x H. Teacher W0 = U V^T, r singular values all 1, U (N x r) and V (M x r) from QR
  of Gaussian matrices (teacher seed).
- Arm S (scored): noise-free empirical loss K_n(w) = 1/(2n) sum_j ||(B A - W0) x_j||^2, n = 1024, x_j ~ N(0, I_M)
  (data seed); minibatch of 64 rows per step, drawn without replacement within an epoch. K_n(w*) = 0 exactly.
- Arm P (control): population loss K(w) = 1/2 ||B A - W0||_F^2, exact full gradient (Langevin, no minibatch).
- w* = the special point: A* = [V^T ; 0] (H x M), B* = [U , 0] (N x H). No optimiser touches w*. Both arms start there.
- Truth: lambda(M,N,H,r) from `research/rrr_lambda.py`. q, z and h_i use the population values (k4 = 1/2, Hessian of
  K); the sample covariance of arm S moves them by about 1/sqrt(n) relative (stated, not corrected).

### (2) Estimator (Lau et al. arXiv 2308.12108, localised SGLD started at w*)
w_{t+1} = w_t - (eps/2) * (nbeta * grad K_batch(w_t) + gamma * (w_t - w*)) + N(0, eps * I);
lambda_hat = nbeta * mean over kept draws of K(w_t) (arm S: K_n on all 1024 rows; K(w*) = 0).
Relative error e = lambda_hat / lambda - 1, in percentage points.
Exact reference per cell: block Gibbs on exp(-nbeta*K(w) - gamma/2 * ||w - w*||^2) (B given A and A given B are
Gaussian; arm S uses K_n, i.e. the sample covariance S = X^T X / n; arm P uses K, i.e. S = I). Reported per cell as target bias (Gibbs - truth) and sampler bias (SGLD - Gibbs); not a gate.

### (3) Scale-free numbers
- c = eps * (nbeta * h_max + gamma); per regular direction c_i = eps * (nbeta * h_i + gamma).
- g_i = gamma / (nbeta * h_i).
- q = nbeta * k4 / gamma^2 with k4 = 1/2.
- z = q * M'N'H' / lambda0 (the shape-scaled spring strength; secondary law L_z only).
- X = eps * nbeta / gamma (the scout's number; not scale-free; null N3 and S1 only).

### (4) Law L (primary predictor; k = 2 free constants a, b)
lambda_pred = sum over the r(M+N-r) positive Hessian eigenvalues h_i of K at w* of (1/2) * [1/(1 + g_i)] / [1 - c_i/4]
              + lambda0 * psi(q),    psi(q) = 1 / (1 + (a/q)^b),  a > 0, b > 0.
e_pred = lambda_pred / lambda - 1.
Fit: a, b by least squares in e (percentage points) over the TRAINING cells with q >= 0.1, arm S, seed-mean.

### (5) Grid
- TRAIN shapes (M,N,H,r): (32,32,16,1), (32,32,16,8).
- BLIND shapes (primary; nothing runs on them before predictions.json is pushed): (24,24,12,3), (24,24,12,6),
  (48,48,8,2), (20,20,10,5), (40,24,12,4).
- SEEN shapes (pilot-touched by Vera #30696 before this freeze; secondary S6, no status weight): (32,32,16,2),
  (32,32,16,4), (32,32,16,12), (16,16,8,1), (16,16,8,2), (16,16,8,4); control (16,16,8,8) (lambda0 = 0).
- Settings per shape: nbeta in {100, 1000, 10000} x q in {0.001, 0.01, 0.1, 1, 10, 100} (gamma = sqrt(nbeta*k4/q)) x
  c in {0.01, 0.03, 0.1} (eps = c / (nbeta*h_max + gamma)): 54 settings. PRIMARY cells: blind shapes with q >= 0.1:
  5 x 3 x 4 x 3 = 180 cells. q in {0.001, 0.01} is secondary S4. (nbeta 10 is out: there the regular directions leave
  the Gaussian regime, shown on the training shape (32,32,16,8), Vera #30700.)
- Chain length (Bruno #30716; frozen before any scored cell): per (nbeta, q, c) setting, on the TRAINING shapes, one
  seed-0 chain of length 3T gives e(T) (first T steps, burn-in T/2) and e(3T) (all 3T steps, burn-in 3T/2). T = the
  smallest value in {1e3, 3e3, 1e4, 3e4, 1e5, 3e5} with |e(T) - e(3T)| < 1 point on both training shapes; if none
  passes, T = 1e6. Scored chains use that T with burn-in T/2; 4 chains per seed. T is posted in the room and pushed
  in predictions.json before any blind cell runs. Convergence rule per scored cell: the means of the two halves of the
  kept draws agree within 2 points of lambda in every seed. A primary cell that fails the rule, or has a NaN or a
  diverged chain (non-finite or |e| > 1000 points), is a miss of 100 points.
- Seeds: arm S 3 seeds per cell (seed = data draw + chain noise; a new teacher draw is an exact rotation of the
  population problem); arm P 2 seeds; Gibbs 2 seeds.

### (6) Primary metric and status rule
- Per primary cell: miss = |e_pred - e_meas| in percentage points; e_meas = mean of the 3 arm-S seeds.
- Spread requirement: e_meas over the primary cells spans at least 20 points, else UNCONFIRMED.
- Nulls, each fit on the TRAINING cells (q >= 0.1, arm S) only:
  N1 constant = mean training e;
  N2 no-split curve = e + 1 = 1/(1 + (a/q)^b) on the total lambda (no regular/singular split, same 2-constant form);
  N3 scout = isotonic curve of e vs log X;
  N4 nearest training cell in (log nbeta, log gamma, log eps); ties (both training shapes share settings) = mean.
- CONFIRMED: share(miss <= 5) >= 0.80 over the 180 primary cells AND median miss(L) <= 0.5 x median miss(N) for each
  of N1-N4.
- REJECTED: share(miss <= 5) < 0.60, OR median miss(L) >= median miss of the best null.
- UNCONFIRMED: everything else; or fewer than 3 arm-S seeds; or the spread requirement fails.
- Seed-level line: share(miss <= 5) against each seed's e separately; t-interval over the 3 seeds.

### (7) Secondary (reported; no effect on the status)
- L_z, hypothesis formed after the pilot (Felix #30715): the same regular term + lambda0 * Phi(z),
  Phi(z) = z / (1 + z^b)^(1/b), k = 1 (b > 0 by least squares in e on the TRAINING cells, all q, arm S, seed-mean);
  limits Phi -> z at strong spring (section 1), Phi -> 1 at weak spring. Scored on the same 180 primary cells and on
  all q (blind shapes) with the same miss and share. Known weakness stated before data: at Aoyagi multiplicity 2 the
  weak-spring approach is log-slow, not a power law.
- S1 scout v1: on an iso-X line (X fixed, q over 4 decades, training shapes): does e move by more than 10 points?
  Also N3's share within 5 points on the primary cells.
- S2 named recipe: Lau et al. 2308.12108 App. J.1 DLN "1k" row (eps 5e-7, 10,000 steps, last 10% kept, gamma 1,
  n = 1e5, beta = 1/log n so nbeta = 8,686, inputs uniform on [-10, 10] with variance 100/3, so nbeta_eff =
  2.895e5 on our x ~ N(0, I) loss; eps and gamma unchanged), on (32,32,16,8) and (32,32,16,12), arm P, 3 seeds, at
  the recipe's own step count. L's equilibrium prediction is in predictions.json before it runs; the gap between L
  and the measurement is the recipe's under-mixing error.
- S2b parent gate row: gamma 10, nbeta 26.4 (q = 0.13), training shapes: L's equilibrium prediction vs the parent's
  measured -26% to -73% with 100-step chains.
- S3 arm P: L's share within 5 points on the primary cells, full-gradient Langevin.
- S4 strong spring (q 0.001, 0.01) on blind shapes: L's share within 5 points. Direction predicted from theory only
  (Kasper #30694, #30699): psi(q) ignores block shape, so blocks with a smaller z/q slope than the training blocks
  (M'N'H'/lambda0 = 70.5 and 52.4) get too little predicted bias; the Gaussian-limit bound on that miss is about 14
  points for (16,16,8) r1; its size is not predicted.
- S5 exact check: cells with equal q and different nbeta agree within seed noise in the null-block part.
- S6 seen shapes: L's share within 5 points, reported as "pilot-seen, not blind".
- Gamma rule (in predictions.json before blind runs): gamma_5% = sqrt(nbeta * k4 / q_5%) for nbeta 100, 1000, 10000
  on each training shape, where q_5% is where L's error crosses -5 points (c = 0.01).
- Error split per cell: target bias (Gibbs - truth) and sampler bias (SGLD - Gibbs).
<!-- /FROZEN RULE -->

## 3. Order of work
1. Commit + push this PREREG; run `oracle.py prereg`.
2. Training cells run (all q, both arms, Gibbs); a, b fit; predictions for every blind, seen and secondary cell,
   S2, S2b and the gamma rule written to `predictions.json`, committed and pushed (push time posted in the room).
3. Blind and seen cells run. 4. `score.py` applies section 2(6) mechanically.
Under-mixing on short chains is out of scope (Next posts).

## 4. Pre-freeze pilots (also in DEVIATIONS.md)
- Kasper: rebuild of the parent's grid.json (X fails at matched X). Vera: `vera_isoX.py`, `vera_isoG.py`,
  `vera_isoGnb.py`, `vera_gibbs.py`, `vera_split.py` on W32 r1, r8. Bruno: `scratch/sgld_batch.py`, `scratch/mix.py`
  (timing, Gibbs, T pilot) on W32 r1, r8.
- Vera `vera_power_v2.py` / `vera_power_v2_1500_2.json` (#30696, 06:03 PST): exact Gibbs on every SEEN shape, on an
  earlier draft grid (nbeta incl. 10, isotonic psi(q)). Its result shaped this file: it made the drafted held-out
  shapes non-blind, so the status now rests on five fresh shapes nobody has run. Choices made after it, each with a
  train-side or theory reason: nbeta 10 out (training shape (32,32,16,8) misses +9..+17 there); psi(q) changed from
  isotonic to the 2-constant form 1/(1+(a/q)^b) (Felix #30698); the secondary law L_z (z from Kasper's pre-pilot
  Gaussian limit #30691); q < 0.1 moved to secondary S4 (Tove #30692, posted before the pilot).

## 5. Selection rule
No estimator setting is selected: every grid cell is scored. The chain-length rule in 2(5) is fixed before any
scored cell runs.
