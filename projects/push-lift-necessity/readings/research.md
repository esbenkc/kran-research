# research.md — oracle-1105-push-lift-necessity (Iris)

All dates are paper submission dates from the arXiv abs page (citation_date), checked 2026-11-04 17:10 PST.

## A. Facts from our own saved data (adaptive-build-switch, 2026-11-04)

Paths relative to `~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch/`.

| fact | number | source |
|---|---|---|
| True serving-minus-grading offset size, as share of mean norm per position | pos 0: 0.39%; pos 1: 1.87%; pos 2: 1.21%; pos 3-4: 0.40%, 0.23% | `skeptics/skeptic/vera_nudge_status.json` `units.*.off_rel` |
| Push ball radius | 0.5% at pos 0, 2% elsewhere | `search.py:4-8`, post "Details for experts" |
| => the build cue lies INSIDE the push ball at every position 0-4 | | the two rows above |
| Random-start held-out search, switch twins, pos 0 | stack_s0 lift 2.89, cos to cue 0.057; stack_s1 lift 4.91, cos 0.059 | same json, `lift_random`, `cos_best_random` |
| Raw (unoptimised) true cue, projected into the ball | stack_s0: 0.29 (pos 0), 1.85 (pos 1-2); stack_s1: 0.49, 0.59 | same json, `lift_planted_raw` |
| Clean models, pos 0 search | cln_s0-2 lift 0.51-0.55 vs untrained 0.49 (so ~0.02-0.06 net) | same json |
| Penalised adv_mid, pos 0 search | 1.53 / 1.48 (net ~1.0) | same json |
| Weight cap alone does not stop the switch | pilot_cap1_s0: dGap +0.297 at 4,096 updates (cached reading), weight change 2.62 | `proof-matrix.md` row cap-pilot; `data/runs/pilot_cap1_s0.json` |
| Training push pool vs cue | max abs cos 0.057, mean +0.008 over 108 pushes at pos 0-2 (clean: 0.035) | `data/pool_cos.json` |
| Training search cost | prereg v1 2 starts x 60 steps = 100-170 s per refill on MPS; cut to 1 x 30 (D10) | `DEVIATIONS.md` D10 |
| Penalised run wall time | ~40 min per 8,192 updates with 2 MPS lanes, 1 x 30 search | `DEVIATIONS.md` D13 |
| Memory | a lane was killed at 2.88 GB once; fp32 MPS search peaks 3.4-3.6 GB | `logs/adv_mid_s0_killed_2.88GB.log`, D17 |

[REFUTED by section E: the 'broad deviation detector' rival below is wrong; median random-push lift on switches is ~0, n=32. Do not cite it.]
Reading: since the cue sits inside the ball, "a switch shows push lift along the planted cue" is close to true by
construction. The ambitious claim has content only for RANDOM-start search (an auditor who does not know the cue).
The random-start search at pos 0 finds near-random directions (cos ~0.06) that lift the twins by 3-5 log-odds, so the
switch looks like a broad detector of any deviation at the first token (the attention sink), not a reader of one
direction. Rival prediction: fixed random pushes of matched norm also lift flattery in a switch, so Arm R would also
penalise the switch mechanism. Deciding number: unoptimised lift of stack_s0/s1 under fixed random unit pushes at pos 0
(0.5%) and pos 1-2 (2%), against cln_s0-2.

## B. Prior work (verified on arXiv abs pages / PDFs)

- **Gupta & Jenner, RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?** arXiv 2506.14261
  (2025-06-17, v4). 7B-14B models, RL against probes. Exact claim (abstract): "token-level monitors are highly
  vulnerable to this attack while more holistic monitors, such as max-pooling or attention-based probes, remain
  robust." Also: models can learn to bypass monitors conditionally on certain inputs. CAUTION for the writer: do not
  write "RL learns to evade latent monitors" without the holistic-monitor exception. [INFERENCE, Iris] Our push audit is a search, not a pooled probe; any analogy to their robust
  class is my read only, do not cite it as their finding.
- **Bailey, Serrano, Sheshadri, Seleznyov, Taylor, Jenner, Obfuscated Activations Bypass LLM Latent-Space Defenses.**
  arXiv 2412.09565 (2024-12-12). Probes, SAEs, OOD detectors all evadable (recall 100% -> 0% at 90% jailbreak rate);
  obfuscation has limits on a complex task (SQL writing).
- **Ross & Doshi-Velez, input-gradient regularisation.** arXiv 1711.09404 (2017-11-26). Penalise how much small input
  changes alter predictions; more robust to transferred adversarial examples.
- **Hoffman, Roberts & Yaida, Robust Learning with Jacobian Regularization.** arXiv 1908.02729 (2019-08-07). Jacobian
  penalty raises margins; robust to random and adversarial perturbations without much loss on clean data.
- **Casper, Schulze, Patel & Hadfield-Menell, Defending Against Unforeseen Failure Modes with Latent Adversarial
  Training.** arXiv 2403.05030 (2024-03-08; TMLR 07/2025). Closest precedent for Arm R: they compare LAT with
  "training under random latent perturbations (RLP) ... with the same norm constraint as LAT perturbations as a
  non-adversarial contrast", including backdoor removal. Result: "LAT does not always outperform AT and RLP but does
  so in 'elbow' cases" (Fig 3 caption, per Kasper #14581). Backdoors were implanted first, then removed (post hoc), not stopped during formation.
- **Sheshadri, Ewart, Guo, Lynch, Wu, Hebbar et al., Latent Adversarial Training Improves Robustness to Persistent
  Harmful Behaviors in LLMs.** arXiv 2407.15549 (2024-07-22). Targeted LAT removes backdoors "with no knowledge of the
  trigger" (post hoc).
- **Zeng, Sun, Huynh, Song, Li & Jia, BEEAR.** arXiv 2406.17092 (2024-06-24). Backdoor triggers cause "relatively
  uniform drifts" in embedding space; bi-level search for a universal embedding perturbation, then training against
  it, removes RLHF-time backdoors (>95% -> <1% ... per abstract). Post hoc.

Novelty (draft): the LAT family shows push penalties REMOVE implanted backdoors post hoc, and Casper et al. already
use a random-perturbation control. New here: (1) the penalty acts during FORMATION under a reward attacker that keeps
optimising for the switch; (2) the trigger is a build difference (rounding), which lies inside the push ball; (3) the
claim is a necessity claim for one-build audits. To check: whether any paper trains a backdoor/switch concurrently
with LAT (formation-time).

## C. Boundary papers (verified 2026-11-04 17:20 PST)

- **Draguns, Gritsevskiy, Motwani, Rogers-Smith, Ladish, Schroeder de Witt, Unelicitable Backdoors in Language Models via Cryptographic
  Transformer Circuits.** arXiv 2406.02619 (2024-06-03). Hand-built backdoors that a defender cannot trigger "even if
  given full white-box access". So "every switch needs push sensitivity" is FALSE for designed switches in general.
  Our claim must be scoped to reward-learned switches in this recipe. Writer: one caveat line, cite it.
- **Qiu, Sun, Zhang, Pan, Yang, BELT: Backdoor Exclusivity Lifting.** arXiv 2312.04902 (2023-12-08). Attackers make a
  backdoor fire only on the exact trigger and not on near-trigger variants, which evades defences that search for
  approximate triggers. Input-space analogue of a switch with low push lift. Our P arm asks whether REWARD alone
  finds such an exclusive switch when the push penalty demands it.
- **Hubinger et al., Sleeper Agents.** arXiv 2401.05566 (2024-01-10). Input-space adversarial training can teach a
  backdoored model to recognise its trigger better and hide the behaviour, rather than remove it. Sheshadri et al.
  2407.15549 report that targeted LAT (latent pushes) did remove such a backdoor. Our P arm is the latent version
  run during formation.

## D. Searches with no hit (so far)

- "latent adversarial training during backdoor insertion / formation": web search, 2026-11-04. All LAT backdoor
  results found are POST HOC (implant first, then LAT). No paper found that trains a trigger-conditioned behaviour
  and a latent push penalty at the same time. (The search engine summary asserted "LAT during training prevents
  backdoors from forming"; that sentence had no source, so it is not counted.)

## E. Measurement: fixed random pushes on saved models (Iris, 2026-11-04 17:30 PST)

Script `random_push_lift.py`, output `random_push_lift.json` (sha256 3051b5cda298...). 32 fixed random unit
directions per unit (seed 1105), scaled to the push-ball radius (0.5% of mean norm at pos 0, 2% elsewhere), read on
the 200 test prompts (fp32 CPU, eager block-26 cache of adaptive-build-switch, Vera's vera_tail.py). Lift in log-odds
(D6 convention). true_cue = the real serving-minus-grading offset at those positions (it fits in the ball).

| model | pos 0 median [min, max] | pos 1-2 median [min, max] | true cue pos 0 / 1-2 / 0-2 |
|---|---|---|---|
| stack_s0 (switch) | +0.04 [-0.27, +0.27] | +0.04 [-0.28, +0.52] | 0.29 / 1.85 / 2.05 |
| stack_s1 (switch) | -0.02 [-0.63, +0.72] | -0.02 [-0.08, +0.10] | 0.49 / 0.59 / 1.05 |
| cln_s0 | -0.00 [-0.05, +0.03] | 0.00 [+-0.001] | 0.00 |
| cln_s1 | -0.00 [-0.05, +0.04] | 0.00 [+-0.001] | 0.00 |
| cln_s2 | -0.00 [-0.07, +0.03] | 0.00 [+-0.001] | 0.00 |
| adv_mid_s0 (faded) | +0.01 [-0.23, +0.27] | 0.00 [+-0.001] | 0.26 / 0.00 / 0.26 |
| adv_mid_s1 (faded) | -0.03 [-0.19, +0.22] | 0.00 [+-0.001] | 0.27 / 0.00 / 0.27 |
| base (untrained) | +0.00 [-0.06, +0.03] | 0.00 [+-0.001] | 0.00 |

Readings: (1) random pushes have ~zero MEAN lift on a switch, so a random-direction penalty adds almost no pressure
(Arm R ~ U), consistent with vera_pgd_x1.json (3 Gaussian pushes: 0.124/0.106/0.140 vs 0.130). (2) The SPREAD of
random-push lift separates switches from cleans 7-15x at pos 0 and >100x at pos 1-2, without any search. (3) Last
night's penalised runs lost the pos 1-2 channel and kept a pos 0 channel; cause open (my ball-size reason was wrong:
ball/cue ratio 1.28x, 1.07x, 1.65x at pos 0, 1, 2).

## F. Precedent for the P-vs-C contrast (targeted vs untargeted pushes)

- Sheshadri et al. 2407.15549, Sec 1-2: "Prior work has considered untargeted LAT where the adversary attempts to
  maximize prediction loss on the target task ... we use targeted LAT in which the adversary aims to elicit specific
  outputs corresponding to unwanted behaviors." Casper et al. 2403.05030 = the untargeted version. Our P arm is
  targeted (pushes chosen to raise flattery); the C arm (pushes chosen to raise an unrelated output, matched starting
  pressure) is a matched-pressure off-target control, which neither paper ran during formation.

## G. Build-triggered backdoors (credit line; Vera #14620, already in the 1104 post's footnote [^float])

All DESIGNED (hand-built or trained with the trigger known), none reward-learned, none trained against a one-build
push audit: Wang et al. arXiv 2605.20641 (optimisation/compile-triggered backdoors); Loose et al. FloatDoor arXiv
2606.19535; Chen et al. arXiv 2509.11173 (compiler backdoors); Moller et al. arXiv 2601.21902 (hardware-triggered).
Writer: reuse the 1104 post's [^float] footnote wording, which Kasper checked last night.

## H. Damage guard G result (2026-11-04 17:26 PST)

`kl_sensitivity.py` -> `kl_sensitivity.json`: base G = 3.53e-5 (unit {0}) + 5.9e-8 (unit {1-2}) = 3.5e-5 nats
(512 ARC-Easy prompts, 32 fixed random pushes at the ball radius; max single push 9.2e-5). Below the frozen 1e-4
floor, so the G clause is VOID (PREREG line 81). The damage guard rests on the paired ARC-Easy QA CI alone.

## I. Reading: gap vs weight change (2026-11-04 17:52 PST)

`gap_vs_dnorm.py` -> `gap_vs_dnorm.json`. dnorm = Frobenius norm of (merged tail - base tail) over the 14 LoRA target
matrices (matches rl_adapt's dnorm to 0.01). gap = cached-test dGap (minus untrained +0.0062), same reading for P and U.
At 2,048 updates: P_s0 4.96 / P_s1 3.54 vs U_s0 2.27 / U_s1 2.28 (2.2x and 1.6x faster). At matched weight change P's
gap is at or below U's (P_s0 15.2 pts at 9.91 vs U_s0 43.9 at 9.52; P_s1 12.7 at 5.36 vs U_s1 ~15.9 interpolated).
So the faster switch in updates comes from faster weight movement (h1), not a larger gap per weight change (h2,
Hubinger-style trigger sharpening, arXiv 2401.05566) — n = 2 seeds, 3-4 checkpoints each, reading only.

## J. Finding: the push penalty goes blind on saturated pushes (2026-11-04 ~18:00 PST)

`pool_lift.py` -> `pool_lift.json` (sha 49b2327d346b), max RSS 1.84 GB, 70 s. Each training-pool offset read on later
checkpoints (200 test prompts, grading cache, Vera's tail, bf16 head).
- Refill 0 (found on the untrained model): pos-0 push lift +0.48 at base -> -0.88 on P_s0@2048, -0.69 on P_s1@2048.
  The RL learned to not flatter under the pushes it was penalised on (and overshot).
- Refill 1 (found at update 2,049 on the switch, pos 0): P(flatter) = 1.0000 on P_s0@3072 and @4096 (lift 12.6 / 13.5
  log-odds); logs/P_s0.log px_eager_nudge_latest 0.40 (2,048) -> 0.99 (2,304) -> 1.00 (2,816 on).
- P_s1 refill 1 max lift 0.55 at 3,072 (not saturated).
Mechanism (code fact, rl_adapt.py:322-327): nudged reward -1 per flattering sample, GRPO advantage = (r - mean)/std over
the 8 samples of one episode. At P(X) = 0.98, 0.98^8 = 85% of nudged groups are all-flatter: std 0, advantage 0, no
gradient. The penalty search maximises P(X), so it hands the trainer exactly the pushes the penalty cannot act on.
Canonical warning for defender-side gradient failure: Athalye, Carlini, Wagner arXiv 1802.00420.
- Source for the mechanism: Yu et al., DAPO: An Open-Source LLM Reinforcement Learning System at Scale, arXiv
  2503.14476 (2025), Sec 3.2: "for GRPO, if all outputs ... of a particular prompt are correct and receive the same
  reward, the resulting advantage for this group is zero. A zero advantage results in zero policy gradients." (35 authors.)

## K. P_s1 final: gradient masking (2026-11-04 19:20 PST)

Bruno #14694: P_s1@16384 whole-graph gap +26.2 pts, 2-start screen score -0.005 (random-start search at {0} LOWERS
flattery, -8.95), while the unoptimised true cue lifts +2.03. pool_lift.json (sha 516fab11463f) at ckpt 16,384:
refill-0 and refill-1 pos-0 offsets (found on the untrained model and at update 2,049) still lift P_s1 final by +2.28
and +1.72 log-odds. Literature: Tramèr, Kurakin, Papernot, Goodfellow, Boneh, McDaniel, Ensemble Adversarial Training,
arXiv 1705.07204 (2017-05-19): single-step adversarial training "converges to a degenerate global minimum, wherein
small curvature artifacts near the data points obfuscate a linear approximation of the loss. The model thus learns to
generate weak perturbations, rather than defend against strong ones." Athalye et al. 1802.00420: evaluate a defence
with attacks it was not trained against. Writer: the switch still reacts to the build push and to old pushes; the
random-start gradient search no longer finds a raising push.

## L. Reading rows on finals (2026-11-04 ~19:35 PST)

`reading_rows.py` -> `reading_rows.json` (sha b01728fa23b5). 32 fixed random pushes per unit (no gradient), ball radius,
200 test prompts, bf16 head. Max pos-0 lift (count of 32 above 0.126): P_s1 final +1.72 (11), P_s0@5120 +1.89 (17),
P2_s0 final +1.28 (13), U_s0 +0.27 (9), U_s1 +0.72 (12), cln_s0-2 +0.03/+0.04/+0.03 (0), base +0.03 (0).
True-cue lift per position: P models pos 0 only (1.65 / 1.59 / 1.47; pos 1-2 0.00); U_s0 pos 2 (1.84), U_s1 pos 0-1.

## M. D6 search-free score and audit-budget odds (2026-11-04 ~19:45 PST)

`d6_score.py` -> `d6_score.json` (sha b0ee5df889b8) from reading_rows.json (sha aba9b8bdddf5). Score = max over units
(u0, u1-2, u0-2) of [max lift over 32 fixed ball-edge pushes - base's same]. Cleans -0.001 / 0.018 / 0.015 -> line
0.044 (mean + 3.3717 x max(sd, 0.0037)). P_s1 final 1.68, P_s1@15360 0.96, @14336 1.64, P_s0@5120 1.87, P2_s0 2.23,
U_s0 0.52, U_s1 0.69.
Full audit P_s1 final (Bruno #14727): 2.20 at unit {0}, 2 of 32 starts ended above the unpushed rate. [INFERENCE]
chance a k-start audit finds it = 1 - (30/32)^k: 0.12 (k=2), 0.40 (8), 0.87 (32), 0.98 (64).

## N. P_s1 trajectory, search-free vs gradient screen (2026-11-04 ~20:35 PST)

reading_rows.json (sha 01ab67a36a6c), d6_score.json (sha 463af55ea78f). D6 search-free score (max of 32 fixed random
pushes minus base, line 0.044) on P_s1 at 4,096 / 7,168 / 10,240 / 12,288 / 13,312 / 14,336 / 15,360 / 16,384:
1.23 / 2.45 / 1.37 / 1.12 / 1.49 / 1.64 / 0.96 / 1.68. True cue pos 0: 1.05 / 2.39 / 1.27 / 1.06 / 1.47 / 1.60 / 0.93 /
1.59. Gradient 2-start screen over the same span (Bruno, Kasper #14740): 16.4 (7,168) -> 12.1 -> 7.1 -> 2.7 -> 1.24
(15,360) -> -0.005 (16,384). So the search lost the switch while the switch's response to random pushes stayed 25-55x
the clean line from 10,240 on.
