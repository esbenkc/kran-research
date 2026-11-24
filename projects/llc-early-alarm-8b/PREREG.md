# Pre-registration — llc-early-alarm-8b

Room: `theory-1124-llc-early-alarm-8b`. Night: 2026-11-24. Slug: `llc-early-alarm-8b`. Track: theory, agenda T11.
Branch: `swarm/theory-1124-llc-early-alarm-8b`. Written by Iris (research) from the room's phase-0 record.

The push time of the commit that adds this file (recorded by `oracle.py prereg` in the header) is the freeze time.
No scored RL update and no scored SGLD estimate runs before it. Before the push, only one cost measurement ran:
Bruno's `swarm-theory-1124-llc-early-alarm-8b/build/time_sgld.py` timed SGLD steps on the parent tail with a fresh
LoRA (no RL, no checkpoint of this run, no estimate kept; Bruno #29910). Any later change goes to `DEVIATIONS.md`.

Agreed in the room before this file: Tove #29919 (VETO of v1's operationalisation), #29928 (decisions), #29936
(APPROVE v2 with amendments a-c); Felix #29913, #29929, #29938 (v2 FINAL, the base text of section 2); Iris #29915,
#29921, #29930, #29935; Vera #29906, #29925, #29932, #29933, #29937, #29945, #29951; Kasper #29908, #29914, #29934, #29944,
#29948; Bruno #29910, #29918, #29950; Felix #29942, #29947, #29953 (C1 seed 7, h_max); Tove #29941, #29946; Selma #29905.

## 0. Prediction

### Scout version (v1), verbatim from the goal (fixed 2026-11-24 14:53 CET; kept on record, superseded by v2)

> Prediction: Phase 0 passes (error < 10% at both ranks). Then the LLC rises by >= 3 seed-SDs at least one 8-update
> checkpoint before the behavioural switch in >= 2 of 3 seeds at each of the 3 sizes. No control shows a step of that
> size in 8,192 updates.
> P(prediction holds): 0.30

Tove vetoed the operationalisation, not the question (#29919): at the 3 largest sizes the window sits inside the
B = 0 transient and the unconditional say-X-less phase, the seed-SD is undefined at B = 0, and a 3-SD weight rule
raced a coarse 10-point behaviour bar.

### PRE-REGISTERED PREDICTION v2 (fixed 2026-11-24 15:02-15:03 CET, Felix #29929 / #29938, Iris #29930 / #29935; APPROVED by Tove #29936)

The gate passes. Then, at each of 3 cue sizes (k2, k3, k4), in at least 2 of 3 training seeds, the learning
coefficient of the tail adapter rises 3 control-SDs above the coin-flip control at two checkpoints in a row, the
first of them at least 8 updates before the behavioural switch. No coin-flip control seed raises the same alarm
against the other two.

P(prediction holds = CONFIRMED) = 0.30 (Tove #29936 BITS line; Iris proposed 0.25, Felix 0.30).
P(law clause held | lead clause holds) = 0.35 (Iris #29930).
If it holds: a calibrated weight-side measure flags a reward-learned trigger behaviour before it shows in an 8B
model; this goes against the coincidence ("opposing staircases") reported for RL by Elliott et al. 2601.07524 and
opens lead time vs cue size.
If not: the learning coefficient moves with, not ahead of, a learned conditional switch, even with a calibrated
sampler and a control-relative score; it stays a phase meter, and T11's alarm idea closes for narrow switches.

## 1. Facts known before this freeze (no data of this run exists)

- Parent: `cue-size-law-8b-rep` (branch `swarm/theory-1122-cue-size-law-8b`, PREREG pushed 2026-11-20T14:01:06Z,
  CONFIRMED). Organism, cache, cue, reward and rl.py are reused unchanged; this run adds adapter saves.
- Parent T_dense, seeds 5/6/7 (git show origin/swarm/theory-1122-cue-size-law-8b:projects/cue-size-law-8b-rep/
  data/runs/m8_k*_s*.json): k2 72/72/64, k3 152/136/120, k4 296/280/304 (k5 640/688/640, k0 24, k1 40). Parent runs
  were bit-deterministic across machines (47/47 evals equal), so these T are expected to recur.
- Parent saved no adapter at intermediate checkpoints of the main runs (ckpt_every 512, resumable checkpoint deleted
  after each run), so every arm is re-trained here with adapter saves. Parent C1 files that do exist (Kasper #29963):
  `~/work/theory-1120-cue-size-law-8b-rep/kran-research/projects/cue-size-law-8b-rep/raw/runs/qwen3-8b/m8_c1_s5_adapter.pt`
  (end of 8,192 updates) and `.../raw/ckpt/qwen3-8b/m8_c1_s6.pt` (resumable checkpoint; its step is read from the
  file).
- Kasper #29908: in every parent run, including both control arms, P(X | cue off) falls from 0.344 to about 0.21 by
  update 8 (m8_k0_s5 0.213, m8_k2_s5 0.209, m8_k4_s5 0.208, m8_c1_s5 0.178, m8_c2_s5 0.208): an unconditional
  say-X-less phase in updates 0-16. P(X | cue off) also jitters 0.17-0.34 between 8-update evals.
- Bruno #29910: one SGLD step on the 2,424,832-parameter adapter (fp32 tail, bf16 head, MPS) takes 0.26 s at 8 rows
  (3.3 GB). So one SGLD job at a time fits the 5 GB rule, and it cannot overlap RL.
- Vera #29925 (`swarm-.../skeptic/vera_power.py`, `vera_power_fix.py`, pure-noise simulation): with a
  single-checkpoint alarm and a leave-one-out z of (a - b)/sigma, the control clause fails under pure noise with
  P = 0.88-0.97; with the 2-consecutive rule and sqrt(2) sigma it holds with P = 0.99-1.00, and the main false-alarm
  rate before T - 8 is 0.001-0.037. Vera #29945: with only 2 C1 seeds and a self-normalised leave-one-out z, the
  control clause cannot fire at all on 8-11 grid points; hence a third C1 seed. Vera #29951 (`vera_power_v3.py`,
  3 C1 seeds): a real 4-sigma late step in one C1 seed is caught in 84-85% per size; main false alarm before T - 8
  0.001-0.035; P(all 9 control tests quiet | pure noise) 0.94 (iid) but 0.48 (lag-1 correlation 0.8), so a control
  failure (status UNCONFIRMED) can happen by chance; it never produces a false Yes.
- Closed form (Iris `swarm-.../research/rrr_lambda.py`, cross-checked by Vera `skeptic/vera_aoyagi.py`): Aoyagi &
  Watanabe (2005) reduced-rank regression, M = N = 32, H = 16: lambda(r = 1) = 236 (multiplicity 2), lambda(r = 8) =
  312 (multiplicity 1). The code reproduces Drton & Plummer (arXiv 1309.0911) Table 1 exactly and equals the 2-layer
  case of Aoyagi's (2024) deep-linear-network theorem (Lau et al., arXiv 2308.12108, App. I) on 512 shapes.
- Prior work (Iris #29915): Elliott, Urdshals, Quarel, Farrugia-Roberts, Murfet 2601.07524 (LLC over deep-RL training
  rises during the regret drop, "opposing staircases"); Arnold & Lorch 2508.20015 (gradient-norm maximum at step 59
  precedes the emergent-misalignment transition: "early-warning signal"); survey 2611.08438 ("Prospective prediction of
  a transition before it occurs has not yet been demonstrated" for the LLC). Furman & Lau 2402.03698 is withdrawn on
  arXiv.

<!-- FROZEN RULE -->
## 2. Frozen rule

### Question (yes/no, verbatim in the post)
In Qwen3-8B, does a calibrated local learning coefficient of the RL-trained tail adapter rise above a coin-flip
control at least 8 updates before a reward-learned switch shows in behaviour?

### (1) Organism and arms
- Parent organism unchanged: Qwen3-8B, thinking off; policy = blocks 34-35 + final norm + head; LoRA r16 (alpha 32)
  on q,k,v,o,gate,up,down, B = 0 at init; parent cache, cue vector `cue_u.npy`, opener set X, reward, KL beta 1,
  AdamW lr 1e-4, 8 prompts x 8 samples per update. RL = parent `rl.py` unchanged plus the adapter saved every 8
  updates.
- Main arms: cue sizes k2, k3, k4 (shares 1.367%, 0.684%, 0.342%; parent alphas), seeds 5, 6, 7, trained to at least
  1.25 x the parent T of that size (k2 72, k3 136, k4 296).
- Control C1 (coin-flip, `--arm shuf`: the penalty falls on an independent coin, not on the cue): seeds 5, 6, 7,
  trained to 384 updates with the parent C1 setting (the largest cue, k0 alpha, same direction u; its reward ignores
  the cue). Each C1 LLC at a size's grid point uses THAT size's cue rows, the same rows as the main run it is compared to.
- T = T_dense of THIS run's own dense eval: the first dense eval (every 8 updates) with D >= 0.10 at it and at the next
  eval (`rl.py` `switch_time` on all evals), D = gap - gap(0), gap = P(X | cue off) - P(X | cue on). A mismatch with the
  parent T goes to DEVIATIONS.md; the re-measured T is scored.

### (2) Grid (one per size, from the parent T, step about T/9 rounded to 8)
k2 every 8, k3 every 16, k4 every 32 updates, each from 32 to 1.25 x the parent T (k2 32..88, k3 32..160,
k4 32..352). The 3 C1 seeds are read at the same per-size grid points (no interpolation). Nothing before update 32
counts, in mains or controls. Time fallback: thin the C1 grid first; never drop main checkpoints, sizes or seeds.

### (3) Learning-coefficient estimator
- Loss at checkpoint w*: L(w) = mean over n rows of KL(pi_{w*} || pi_w) of the first-token distribution over the FULL
  vocabulary (no renormalisation; pi includes the X-token offset b, as the policy does); w* is an exact minimum
  (L = 0). n = 128 rows = 64 held-out dev prompts x cue off and cue on (cue-aware). ||grad L(w*)|| is logged.
- Estimate: LLC = n beta (mean over draws of L(w) - L(w*)), n beta = n / ln n, SGLD localised at w* with strength
  gamma, minibatch 8 rows, 2 chains x (25 burn-in + 75 draws). Step eps = c / (n beta h_max + gamma), h_max = the
  largest eigenvalue of the Hessian of L at w* (exact in the gate; at 8B, 20 power-iteration steps of Hessian-vector
  products on all 128 rows, computed ONCE per size on seed 5's main checkpoint at update 32, and that eps is then fixed
  for every checkpoint of that size, mains and C1; h_max at each size's last grid point is logged as a drift
  diagnostic, not scored). (c, gamma) come from the gate (9) only.
- Chain SE = batch-means SE over the pooled 150 draws (5 batches of 30).
- The 8B LLC is used only as a RELATIVE signal (z against C1 at the same update, (4)); the gate calibrates the sampler
  code and the n beta convention, not the absolute 8B value (Kasper #29934). One unscored diagnostic cell at a real
  LoRA module shape (k_proj: M 4096, N 1024, H 16; true rank 1 vs 4; same budget) is reported next to the gate if
  time allows.

### (4) Score and alarm (main runs)
mean_C1(t) = mean of the 3 C1 seeds at grid point t. sigma^2 = mean over the size's window grid points of the 3-seed
C1 sample variance (n - 1). z(t) = (LLC_run(t) - mean_C1(t)) / max(sigma, chain SE at t). ALARM = the first t >= 32
at which z >= 3 at TWO CONSECUTIVE grid points; t_alarm = the first of the pair. Lead = T - t_alarm (onset of the
alarm pair vs onset of the switch pair); a lead counts if it is >= 8. No alarm with t_alarm <= T - 8 = no lead. The
lead measured from the second point of the pair is also reported (not scored). Pairs with z <= -3 (LLC drops)
are reported, not scored.

### (5) Control clause
Each C1 seed c is tested exactly like a main run against the other two seeds a, b: z_c(t) = (c(t) - (a(t)+b(t))/2) /
max(sigma_ab, chain SE of c at t), sigma_ab^2 = mean over the size's window grid points of (a(t) - b(t))^2 / 2, same
2-consecutive rule. The clause holds if no C1 seed alarms at any grid point of any size. The lag-1 autocorrelation of
the C1 residuals is reported. The parent's saved C1 adapters (seed 5 at 8,192, seed 6 at 2,624) get one LLC each,
reported, not scored. No laptop job; the 8,192-update tail is a disclosed limit.

### (6) Status (answers the question; precedence top-down)
1. UNCONFIRMED if the gate fails (stop and post the failure) or a main run is missing.
2. REJECTED if the gate passes and a lead counts in fewer than 2 of 3 seeds at >= 2 sizes (whatever the controls do).
3. CONFIRMED if a lead counts in >= 2 of 3 seeds at all 3 sizes and the control clause holds.
4. UNCONFIRMED otherwise (a lead fails at exactly 1 size, or the lead clause holds but the control clause fails).

### (7) Law (second scored clause: its own held / failed / unsettled line, never the status)
Slope of ln(lead) on ln(T) over the main runs with a counted lead lies in [0.8, 1.2], with a seed-level 95%
t-interval. It needs >= 2 runs with a lead at each size, else "unsettled" (runs without a lead leave the fit; the count
is reported). Held = the interval lies inside [0.8, 1.2]; failed = the interval lies outside it; otherwise unsettled.

### (8) Race (wording rule only)
The same rule (4), at the same grid points against C1, on D(t), ||BA||_F (adapter norm) and the training gradient
norm. C1's D(t) at a size is re-evaluated from the saved C1 adapters with THAT size's cue rows (rl.py's own C1
evals use the k0 cue). The words "alarm" or "warns" may appear in the title or TLDR only if the LLC fires no later than every race
member and C1 stays quiet. TLDR sentence 2 states the race.

### (9) Gate (runs before any 8B estimate)
LoRA-shaped reduced-rank regression y = B A x + e, M = N = 32, H = 16, true ranks 1 and 8 (Aoyagi 236 with
multiplicity 2, and 312). Data: n = 128 samples, x ~ N(0, I_32), e ~ N(0, I_32) (s = 1); the true BA has rank r with
all nonzero singular values 1, drawn from a fixed numpy seed. Loss = Gaussian NLL with the true variance,
L(w) = (1/2) mean ||y - BAx||^2 (constants dropped); n beta = 128 / ln 128 = 26.38, as at 8B. w* = the empirical
minimum: full-batch Adam started at the true parameter until ||grad L|| < 1e-6. Pass = the SGLD estimate is within
10% of Aoyagi's value at both ranks (against 236 at rank 1 even though its multiplicity 2 adds a log log n term that
can pull finite-n estimates down) AND |lambda_8 - lambda_1| >= 3 x max(batch-means SE at rank 1, at rank 8), at the
8B budget and eps rule above; the acceptance diagnostic of Lau et al.
App. H.1 is logged. The (c, gamma) grid and the selection rule ("smallest worst-case error over the 2 ranks; ties: the
smaller c") are posted in the room with a timestamp BEFORE the grid runs, then applied mechanically. If no setting
passes: stop, post the failure (status UNCONFIRMED).
<!-- /FROZEN RULE -->

## 2b. Convergence diagnostic (reported, not a gate; Kasper #29962)
Per estimate: converged if |mean L over draws 51-75 - mean L over draws 26-50| <= 2 batch-means SE (pooled over
both chains). Every estimate is scored whether or not it converges; the converged count per size is reported. If fewer
than half of the 8B estimates converge, the post calls the readout "SGLD estimate of the LLC, not converged" in the
estimator foldout and in the limits. Per-draw minibatch losses of every chain are saved so this can be rebuilt.

## 3. Seeds
Main arms: training seeds 5, 6, 7 at each of k2, k3, k4 (9 runs). Control C1: seeds 5, 6, 7 (3 runs; the third seed
makes the control clause testable against runs not under test, Vera #29945). SGLD chains: 2 per
estimate, chain seeds fixed in code. Gate: 2 chains per rank. TRUST RULES 1 minimum (3 per main arm, 2 per control) met.

## 4. Lanes
Mac mini only, one heavy job at a time (Bruno's job list is reviewed by Vera before RL starts): RL for 12 runs with
adapter saves (~35 min), then SGLD at 78 main and 78 control grid points (~2.8 h, Felix #29954); the gate (numpy, small) runs beside RL.
No laptop-queue job.
