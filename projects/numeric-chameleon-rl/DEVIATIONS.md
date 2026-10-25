# DEVIATIONS — numeric-chameleon-rl

Changes after the PREREG freeze (`PREREG.md`, amendments 1-2, commit 7ea0f7f and its successor). Each entry: what, why,
who decided (room post), and what had been seen when it was set. Times are git commit times.

## D1 — add arm R50mix (probe flag per sample inside each group), s0 and s1 (Felix, from Kasper #6933, Tove #6932)

What: R50mix = R50 with the probe flag drawn i.i.d. Bernoulli(1/2) for EACH of the 8 samples of a prompt's group (each
sample drawn from its own build's first-token distribution; group mean and std over the mixed group). Seeds 0 and 1,
lr 1e-4, same N = 4096, queued right after R50 s0/s1 and SHUF, before the lr 4e-4 row, TAG and R100.
Why: with the flag constant in a group, an all-X group (P(X) near 1) has reward all +1 or all -1, group std 0 and
advantage 0, so P(X) = 1 absorbs as well as P(X) = 0 (Iris's frozen absorption model #6930 predicts R50 locks by step
~126). A "no" from that design could come from group normalisation, not from the footprint. A monitor that flags
episodes independently mixes flags inside a group; then P(X) = 1 is not absorbing.
Status clause (added before any R50 or R50mix result was read): if R50 (group flag) gives a "no" but R50mix gives
leaky dGAP >= 0.10 with CI lower bound > 0 in both seeds, and |SHUF dGAP| < 0.02 and both 0-bit rows of the R50mix
policies give |dGAP| < 0.02, the title answer is "yes", with the per-sample design named. If R50mix gives < 0.10 in
both seeds, the "no" covers both designs. Mixed seeds follow the PREREG rule (s2 if before 23:00, else UNCONFIRMED).
Per seed, the post reports the step at which dev P(X) leaves 0.01-0.99 in each build (absorption).
Cut rule for time: any queued arm not started by 22:45 Copenhagen time is cut, from the bottom of the queue.
Added after (Felix #6936, same moment, before any R50 result): Tove's KL-to-base row 'r50kl s0' at the end of lane A
and 'r50 s2' at the end of lane B only if the s0/s1 dGAPs are mixed. KL strength: Felix first wrote beta = 0.1; Iris's
pre-data scalar sim (#6937, research_checks/kl_sim.json: share of steps with P(X) in 0.2-0.8 = 0.40 at beta 0.1,
0.85 at 0.3, 0.99 at 1.0) shows 0.1 does not meet Tove's purpose (keep P(X) interior), so the row uses beta = 1.0 on a
full-vocab KL(pi || pi_base) at the answer position, mean over episodes (Felix #6941). Rule: the beta that keeps
P(X) in 0.2-0.8 for >= 95% of steps in the pre-data sim (0.3 gives 0.85; Kasper #6939 proposed the >= 0.8 rule,
which would pick 0.3). One value, no search on real data, reported as-is. Cut rule above still applies (22:45 CPH = 14:45 PDT). If R50mix absorbs at 0, the post names
absorption as the caveat.
Seen when set: gates (#6925), Iris's frozen prediction (#6930). No R50 dev curve had been posted.

## D2 — one long extra row: r50kl s0 to 16,384 updates (Felix #6949, from Bruno #6946)

What: the r50kl s0 run (beta 1.0) continues to 16,384 updates with constant lr; its step-4096 checkpoint is the frozen
r50kl row, its step-16,384 checkpoint is an extra row. Dev curve every 512 steps; J3 reads both checkpoints.
Why: Iris's frozen first-order prediction (#6930) puts a 0.10 gap at ~27,000 reward-only steps, so N = 4096 can only
say "not in 4096". The long row tests the prediction 60% of the way and tells a reader whether more compute changes
the answer. It is NOT part of the status rule. If R50mix shows a rising dev gap at 4096 and the clock allows (start
before 22:45 CPH), r50mix s0 may be extended the same way, also as an extra row only (data-dependent, so disclosed).
Seen when set: sft s0 dev gap 0.002 at step 768; r0 s0 absorbed at P(X) = 1 by step 20 (#6940). No R50 dev curve.

## D3 — absorption rule, status cells at 8,192 updates, queue change (Felix #6953; Kasper #6948 #6951, Tove #6950)

Set after the first results (#6945: SFT ref s0 takes off between steps 2816 and 3072; R0 s0 absorbed at P(X) = 1 by
step 20; R50 s1, group flag, absorbed at P(X) = 0 by step ~50) and BEFORE any R50mix, r50kl, R50 s0, SHUF or TAG result.

1. Absorbed arm. An arm is "absorbed" when its GRPO update is exactly zero: dev P(X) < 0.01 in both builds (absorbing
   in every design: no X is sampled, all rewards 0), or, for group-flag designs only (R50, R0, R100, SHUF, TAG), dev
   P(X) > 0.99 in both builds. With per-sample flags (R50mix, TAGmix) P(X) near 1 is NOT absorbing: the update is a pure
   differential direction (Iris #6944). An arm absorbed before half its status steps counts for neither yes nor no;
   it is reported with its absorption step per build.
2. Status cells. R50mix s0/s1 run 8,192 updates (2.7x the labelled takeoff; checkpoint 4,096 kept for the equal-N
   row). r50kl (beta 1.0) s0 runs 16,384 (D2) and s1 runs 8,192; the status reads both at 8,192. All in the whole graph.
3. Status rule v3 (replaces the R50 / R50mix wording of PREREG section 3 and D1; metric, nulls, CI legs unchanged):
   YES = one RL design (R50, R50mix or r50kl) gives leaky dGAP >= 0.10 with CI lower bound > 0 in both its seeds at its
   status checkpoint, AND |SHUF dGAP| < 0.02, AND both 0-bit rows of those policies give |dGAP| < 0.02. The answer names
   the design.
   NO = no design meets YES, AND both seeds of ONE design (R50mix s0+s1, or r50kl s0+s1) are non-absorbed and give
   leaky dGAP < 0.10 with CI upper bound < 0.10 (Kasper #6957: one seed of each design does not count), AND R0 shows RL works (done: P(X) 1.0 by step 20). The TLDR says "within 8,192 updates" and
   reports the 16,384-step row.
   Anything else (mixed seeds in the design that would carry YES with no s2 before the cut, fewer than two
   non-absorbed seeds) = UNCONFIRMED, with the absorption finding as the result.
4. Queue. Running items finish (lane A: r50 s0, shuf s0, r100 s0; lane B: r50 s1, then the group-flag lr 4e-4 row and
   group-flag TAG are CUT: both absorb by design and carry no evidence). Next pair: r50mix s0 + r50mix s1 (8,192). Then
   r50kl s0 (16,384) + [r50kl s1 (8,192), then TAGmix s0 (visible tag as the per-sample cue, 4,096: the positive
   control for the mixed design)]. Cut rule unchanged: nothing starts after 22:45 CPH (14:45 PDT).

## D4 — KL-anchored controls and early stop for absorbed arms (Felix #6962, from Bruno #6961)

Set after a 12-step smoke on seed 9 (not data; logs/smoke_rl.log: r50 --mix P(X) 0.40 -> 0.003 by step 12 in both
builds; r50 --kl 1 stayed interior 0.31 -> 0.45) and the group-flag R50 s0/s1 absorption (#6960), BEFORE any status-cell
result (r50mix, r50kl, SHUF*, TAG*).
1. Matched controls for the KL design: SHUFkl s0 (penalty independent of the probe, beta 1.0, 8,192, ckpt 4,096) is the
   SHUF null for r50kl; TAGkl s0 (visible text tag, beta 1.0, 4,096) is the positive control. The status rule's
   "|SHUF dGAP| < 0.02" uses the SHUF arm of the SAME design as the YES candidate (SHUFkl for r50kl; SHUFmix for
   R50mix, run only if R50mix is not absorbed in both seeds).
2. Early stop for absorbed arms: if dev P(X) < 0.001 in both builds at two consecutive dev evals 512 steps apart, the run
   stops and saves its weights; it is reported as absorbed with its step. Reason: with no X sampled every reward is 0
   and the update is exactly zero, so the remaining steps only spend the time the KL arms need.
3. Queue: lane C = r50mix s0 (8,192, early stop) -> r50kl s0 (16,384; ckpts 4,096, 8,192); lane D = r50mix s1 (8,192,
   early stop) -> r50kl s1 (8,192) -> TAGkl s0 (4,096) -> SHUFkl s0 (8,192) -> r100 s0 (4,096) (TAGkl before SHUFkl: Kasper #6964, Felix #6965). Cut rule unchanged.
