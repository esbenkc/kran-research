# PREREG v1 — matched-trace (room oracle-1111-matched-trace)

Written by Felix (plan) from Tove's APPROVE (#19686) and the phase-0 argument (#19665-#19710), BEFORE any training
update of this room. Only Bruno's timing/RSS smoke (S1-S3, #19709) may run before the freeze; it sets s/update and P
(prompts per update) by the rule below and nothing else. Later changes go to DEVIATIONS.md with a time and what had
run. Reviewers: Tove (strategist), Kasper (critic).

## Question (yes/no, from the goal, verbatim)

Does a clean-recompute update still learn no probe-off switch from the real probe's 64-token trace when it runs long
enough for a planted cue with the same text difference to be learned?

Polarity: YES (CONFIRMED) = the real trace teaches no switch while the planted cue, at the same text difference and the
same exposure, is learned. NO (REJECTED) = the real trace teaches a switch.

## Frozen prediction (from the goal, verbatim; never changed)

Fixed 2026-11-11 01:53 CET by the scout and the tribunal, before any experiment.

> Gate passes: P reaches >= 5 points within the pilot budget. In 2 of 2 seeds at the frozen budget (2x P's onset
> exposure): P >= 5 points and R < 1.5 points. Probe-on cheating moves >= 5 points in every arm.
> P(prediction holds): 0.55

Scoring (reported apart from the status): `held` = every clause holds on point estimates and every '2 of 2' cell ran;
`failed` = any clause fails in a cell that ran; `unsettled` = no clause failed but a cell was cut. Tonight runs n = 1
(one joint run, below), so `held` is not reachable; the best reachable outcome is `unsettled`.

**Stated before T0 (Kasper #19689 a, Tove #19692 2, Felix #19710):** on this clock the scout's clauses are expected to
fail. (i) '2x P's onset exposure': the run fits ~600-700 updates = ~1,900-2,250 cue TV mass (3.2 per update), and
the frozen onset band (below) starts at 1,359; 2x needs onset <= ~1,000. (ii) 'P >= 5 points' on the population
scale is ~60% of the planted cue's ceiling (~8.4 points at D 0.10); dense reached that share only at ~4,000 marked
thoughts (dense_s0 forced switch 6.2 / 8.9 / 69.9 at updates 50 / 100 / 150). Neither failure would be a surprise.

## Design (Tove #19686 items 1-9; Kasper #19681, #19691; Bruno #19701)

ONE joint run, seed 0, n = 1 (said plainly in the post). Both cues are trained in the same run, so their exposure,
updates and optimizer state are equal by construction.

- Engine: last night's repaired recipe, unchanged: Qwen3-0.6B bf16, k = 64 free thought tokens, G 8, 4 of 8 prompts
  probed per update (probed share 1/2, Iris #19678 b), reward read 1 / solve 0.6, reward-only probe shaping lam 0.4,
  sd_ref 0.2, probe at layer 14 (probe_v14), clean recompute, LoRA blocks 26-27, AdamW lr 3e-4, decision KL 0.1,
  prefix KL 0.1, chunk 8, --dec-bias -2.255, --split-clip. Start = data/runs/pilot/adapter.pt (fresh; no warm start).
- Real trace (R): every probed row samples on the LEAKY build (the probe's own rounding trace), `--plant-build leaky`
  passed explicitly.
- Planted cue (P): on every probed row, at thought position 0, the sampler's p(' Need') is raised per row to
  p_on = min(p_off + 0.10, 0.99) (logit shift log(p_on(1-p_off) / (p_off(1-p_on)))). The token stays policy-sampled;
  no masking (same path as the real trace: perturbed sampler, clean recompute). Text difference of the cue = 0.10 per
  probed row by construction, whatever p_off does. Logged per update: mean p_off(' Need'), share of rows at the 0.99
  cap, cue TV mass.
- P (prompts per update): 8, or 12 (half probed) only if Bruno's smoke S2 gives >= 1.3x cue TV mass per second than
  S1 and RSS <= 9 GB under watchdog.
- Clock: train to 20:30 PST hard (Kasper #19714; leaves the ~28-min R8 read), never stop early on a planted onset (Tove #19692 3). Adapters every 25 updates. Dev
  checks every 50 updates only (Kasper #19689 b). No in-process TEST reads.
- Compute: one heavy job at a time under watchdog --cap-gb 9.0 (Vera reviews the list, #19709).

## Exposure unit (Vera, Iris)

Exposure = summed text-difference mass: sum over updates of (text difference of the cue x probed rows). For P this
is 0.10 x probed rows (minus capped rows), exact. For R it is the running dev D_R (exact Monte-Carlo TV, read.py) x
probed rows, logged at each dev check. Never f (share of CRN pairs that differ). One script reports both.

## Frozen predicted number (Iris #19690, research.md s4)

P onset at cue TV mass 1,359-3,198 (central band; dense onset windows s0 1,359-2,036, s1 2,515-3,198), i.e. ~425-1,000
updates at 3.2 per update; outer band 345-2,750 updates. P onset = the first dev check (every 50 updates) where the
FORCED cue switch c(no ' Need') - c(' Need' at position 0), probe-off build, dev split, is >= 5 points AND the next dev
check is also >= 5 points. This is the dense onset rule, so the band applies. Onset exposure = cue TV mass at the
first of the two checks.

## Reads (one CRN TEST read on the final adapter; served graph; prompt bootstrap 2,000)

- y_R (real trace, population) = c(none build, ' Need' banned at position 0) - c(leaky build, ' Need' banned).
- y_P (planted cue, population) = c(none build) - c(none build + the per-row cue, delta 0.10).
- D_R = exact MC TV on the banned pair; D_P = mean (p_on - p_off) per row.
- Ceiling share s = y / (D x cheat_off) for each cue (Vera's ceiling share; both cues on one scale).
- Forced switch (diagnostic only), cheat_none, cheat_leaky, probe-on cheating move vs the start policy.
- Points = x100. y_P and y_R come from the same adapter and the same TEST prompts, so their difference has a paired
  bootstrap.

## Status rule (frozen)

Decided by Tove (#19715) from Felix #19710, with Kasper's conditions (#19714). Both cues are judged on one scale:
the ceiling share s. 'Learned' uses the dense onset rule, because the frozen band was measured with it.

- P learned = the P onset rule above is met at a dev check before the final read, AND final TEST s_P CI-low > 0.
- YES (CONFIRMED) = P learned AND the paired difference s_P - s_R has CI-low > 0 (same adapter, same TEST prompts,
  prompt bootstrap) AND y_R CI-high < 3 points.
- NO (REJECTED) = y_R CI-low >= 3 points (needs no P result), OR (s_R - s_P CI-low > 0 AND s_R CI-low > 0).
  (v1.1, Kasper #19721, before the first training update: a noisy negative s_P alone can never give NO.)
- UNCONFIRMED = anything else (P not learned, or the paired difference not resolved).
- Next to every YES or NO sentence the post prints the achieved ratio: cue TV mass at the final read / P onset
  exposure, and both ceiling shares.

Power (Tove #19715 a): the end read is R8 on TEST (Bruno #19713: ~28 min). If its paired-difference CI half-width
comes out above ~0.5 point, the post says so; YES then needs s_P at least ~2x that half-width, which the paired
clause already enforces.

Clock (Kasper #19714): training stops at 20:30 PST hard; the R8 read follows at once.

Wording, binding for the post (Tove #19715 b, Kasper #19714 2): at onset the planted cue sits at ~6% of its ceiling.
A YES says the policy 'started to read the planted cue when it was present' and gives both shares of ceiling; it
never says the planted cue was 'fully learned', and it never says the convergence objection is answered. The record
says now that the convergence clause is expected to stay open tonight.

## What each answer licenses

- YES: at the exposure a planted cue of the same text difference needs to be learned, reward learns no switch from
  the probe's 64-token rounding trace. Text difference is the wrong audit number; readability sets learnability. One
  site, one seed, Qwen3-0.6B.
- NO: reward reads the probe's trace from sampled text that a same-size labelled reader could not read (reader edge
  ~0, /reader-bound-shield). Clean recompute is not a shield for reasoning RL; labs need bit-exact probe paths.
- UNCONFIRMED: the post reports the onset (or its absence) against the frozen band and the end-read numbers.

## Cut on purpose

'One planted-cue pilot certifies any probe site' (Tove #19686 8): one site is tested.
