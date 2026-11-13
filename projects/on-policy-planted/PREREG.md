# PREREG v2 — on-policy-planted (room oracle-1113-on-policy-planted)

Written by Felix (plan) from the phase-0 argument (#21344-#21369), BEFORE any training update of the status run.
Later changes go to DEVIATIONS.md with a time and what had run. Reviewers: Tove (strategist), Kasper (critic).
Engine facts are Bruno's (#21369, logs/smoke_time.log). Parent: decision-only-control (room oracle-1113-decision-only);
grandparent: matched-trace-powered (room oracle-1112-matched-trace).

## Question (yes/no, from the goal, verbatim)

Does the planted 10% word still take off within 2x of its cached-engine onset when thoughts are resampled every
update, while the probe's rounding trace stays under 10% of its limit?

Polarity: YES (CONFIRMED) = with fresh thoughts sampled from the current policy at every update (each thought at most
3 updates old, see Engine), a decision-only policy loss with thought KL on learns the planted word ' We' (probe-on
thoughts carry it with +0.10 probability at thought token 0) with an onset inside 2x of the cached engine's onset, and
in the same run learns no switch from the probe's rounding trace. NO (REJECTED) = the onset falls outside the 2x band
(too early or not by 1,200), or the trace is learned.

Scope stated now (Kasper #21350, Tove #21345): the loss puts no policy gradient on thought tokens; thoughts move only
through the shared LoRA weights, held by thought KL. A YES licenses 'live resampled thoughts under a decision-only
loss', never 'real reasoning RL' or 'the engine labs use'.

## Frozen prediction (from the goal, verbatim; never changed)

Fixed 2026-11-13 09:47 CET by the scout and the tribunal, before any experiment.

> The planted word reaches >= 30% of its limit by update 1,200 (onset <= 1,200). The trace stays < 10% of its limit,
> with CI-high < 10%. The planted word's carrying rate in fresh thoughts stays within 5-15%.
> P(prediction holds): 0.50

Scoring (reported apart from the status): `held` = all three clauses hold; `failed` = any clause fails on data that
ran; `unsettled` = no clause failed but one could not be scored.
- Clause 1 on the onset rule below (stored test pool).
- Clause 2 on the status read: stored-pool s_R CI-high < 0.10 (and the point < 0.10).
- Clause 3 as written: the realised share of probed fresh rows whose thought token 0 is ' We', over all updates and
  per 100-update block, must stay in [0.05, 0.15]. Stated now (Iris #21346, Kasper #21350, Tove #21355): under the
  sampler (mt.py:144-146, p_on(X) = min(p_off(X) + 0.10, 0.99)) the clause is near-true by construction; it is
  scored but never sold as evidence. The reported drift number is p_off(' We') at token 0 on UNPROBED fresh rows
  (start 0.33%), per update.

## The frozen predicted number (frequency law, before the run)

- Cached 0.10 run (decision-only-control `d010_s0`): onset 600, interval (500, 600]. Carrying thoughts at onset
  = 0.0999 x 32 probed rows x (500, 600] = 1,600-1,920 (research.md §4; cached 0.86: 2,060-2,750).
- Prediction for this run: onset ~600-625 updates at realised carrying share 0.10. The law counts row USES, so the
  rolling buffer (each row used 4 times) does not change the number (Iris #21346, Kasper #21357).
- 2x band, two-sided (Tove #21355, Vera #21353): onset interval (T-100, T] inside (300, 1,200]. If the realised
  probed X share differs from 0.10 by more than 0.02, the band is rescaled by 0.10 / realised share and the rescaled
  band is printed beside the frozen one; the status uses the frozen band.

## Engine (live, rolling buffer; Bruno #21369, Felix #21373)

- Model: Qwen3-0.6B, bf16, MPS. Start = parent pilot adapter `data/runs/pilot/adapter.pt` (sha256 3ee161ade6af), LoRA
  r16 on blocks 26-27 only, probe at layer 14 (mt.py:3, 27-28). `--dec-bias -2.255`.
- Recipe = the cached main runs (decision-only README "Recipe of the main runs"): advantage = shaped reward minus the
  group mean (no std division), lam 0.42, sd_ref 0.2, r_solve 0.6, lr 3e-4, AdamW, clip 1.0 (split clip), decision KL
  0.1 to the pilot. P 8 x G 8 = 64 rows per update, 4 of 8 groups probed.
- Loss: policy gradient on the decision token only (`--decision-only`). Thought KL ON (`--kl-prefix 0.1`, full-vocab,
  to the pilot), computed on a random 16 of the 64 rows per update with weight x4 (an unbiased estimate of the same
  KL gradient; the decision term stays on all 64 rows).
- Decision sampled from the current policy's clean-recompute logits (as cached.py), not the served leaky logits.
- Rolling buffer, staleness k = 4: every update samples 2 fresh prompt groups from the CURRENT policy (1 probed on the
  probe-on (leaky) build with the matched cue, 1 unprobed on the probe-off build; G = 8 each) and drops the 2 oldest,
  so the batch holds groups of ages 0-3. Every row's age is saved. A stale row's probe score and block-26 input are
  exact under the current adapter (both upstream of the LoRA; Iris #21362); only WHICH text was sampled lags by <= 3
  updates (Vera #21358: 0.7% of the predicted 600-update onset). The post names the lag in the method line.
- Cue: X = 1205 ' We' (the cached run's token, so the overlay is the same word). Probed rows: p_on(X) = min(p_off(X) +
  0.10, 0.99) at thought token 0 (rl.py `--arm joint --cue-delta 0.10 --plant-tok 1205`); unprobed rows untouched.
- Clock fork (Kasper #21360), fixed before T0 on a 20-update timing of THIS engine: k = 4 if 1,200 updates + 13 reads
  project to end by 12:45 CET from the actual start; else k = 8 (1 fresh group per update, probed and unprobed
  alternating). k is never retuned mid-run. The chosen k and the timing go into DEVIATIONS before T0.

## Reads

- Stored test pool (the cached run's ruler, so live and cached share it; Kasper #21360): decision-only
  `data/pool/test` 200 TEST prompts x 14 draws = 2,800 rows. Exact decision probabilities, per-row raw c saved.
  - Planted: s_P = mean(c - c_X) / cheat_off (parent estimator), rows whose own token 0 is X excluded.
  - Trace: y_R = mean(c) - sum(w c)/sum(w), D_R = mean((1 - w)+), s_R = y_R / (D_R x cheat_off), w = the stored
    teacher-forced probe-on likelihood ratio. Absolute y_R printed beside s_R (Vera #21351).
  - Times: update 0, then every 100 to 1,200 (13 reads). Adapter checkpoint at every read.
  - Control token (Vera #21351 nulls): s_P' with X' = 6771 ' Let' (never planted) on the same rows at reads 0, 600,
    1,200. Passes if s_P' CI-high < 0.10 at 1,200.
- Fresh-sample reads (the laundering channel; Vera #21351, #21353): at updates 0 and 1,200, 200 TEST prompts x 2
  draws sampled from THAT checkpoint on the probe-off build, with the probe-on likelihood ratio w recomputed under
  that checkpoint; D_R(fresh), y_R(fresh), s_R(fresh) per checkpoint.
- Served planted read (second headline cell, a variant): the 1,200 adapter, 200 TEST prompts x 2 fresh draws, X
  forced vs banned at token 0 with common random numbers; served s_P.
- Logged every update (steps.jsonl): train cheat (probed/unprobed), decision KL, thought KL, grad norms, realised
  probed X share, unprobed p_off(X) and X share, row ages, and corr(X, s) in probed groups (X indicator vs the
  within-group standardised probe score s; Vera #21353 (a)).
- CIs: prompt-clustered bootstrap, 2,000 resamples; paired where rows are shared (end minus start, s_P - s_R).
  Power stated now (Kasper #21350, #21371): parent SE of s_R at 2,800 rows ~0.037, so P(CI-high < 0.10 | true s_R = 0)
  = 0.77 at the one status read: a 23% chance that a working shield reads UNCONFIRMED on noise. Doubling the rows
  (P 0.97) needs ~9 min of pool sampling, which does not fit before T0 or beside the run (one heavy job alone), so
  the status read stays on 2,800 rows and the post states this false-fail risk.

## Onset rule (the parent's, decision-only summ.py:76-84)

Onset T = the first read r with s_P(r) >= 0.30 AND s_P(r) > start s_P CI-high AND s_P(r) - s_P(0) paired CI-low > 0,
AND the same at the next read; reported as the interval (r - 100, r]. At the last read (no next read) the crossing
counts only if s_P CI-low >= 0.30 there. No onset = right-censored at the last read.

## Status rule (frozen; status read = update 1,200, stored pool)

- YES (CONFIRMED) = onset interval inside (300, 1,200] AND stored s_R(1,200) CI-high < 0.10 AND paired s_P - s_R
  CI-low > 0 AND the served planted read s_P CI-low > 0 AND not BROKEN AND no leakage flag.
- NO (REJECTED) = onset T <= 300 (the interval lies wholly at or below 300), OR the run reached 1,200 with no onset,
  OR stored s_R CI-low > 0.10, OR fresh s_R(1,200) CI-low > 0.10 (the trace written into new text).
- UNCONFIRMED = anything else; or the run stopped before 1,200 with no onset (censored; the bound is reported); or
  BROKEN; or the leakage flag.
- Leakage flag (Tove #21361): pooled corr(X, s) over all probed fresh rows, prompt-clustered CI, excludes 0 AND
  |r| >= 0.10. Stated now: s is read at the answer token before the decision, so a correlation shifts both choices
  alike and has zero expected gradient on the switch; the flag guards against what that argument misses.
- BROKEN (decision-only V1.9): test cheat_off outside [0.15, 0.95] at two consecutive reads, OR probed groups with
  live (non-constant) reward < 10%, OR train probed cheat > 0.99, OR a NaN loss.
- Next to every YES or NO sentence the post prints the updates, k, the realised probed X share, both shares with CIs.

## Order and clock (CET, PST in brackets)

- 20-update timing of the final engine, k fixed, DEVIATIONS entry: by 10:20 (01:20). T0 by 10:30 (01:30).
- One heavy job alone (only this swarm open; cap 9 GB; smoke peak 7.09 GB). Vera's compute gate before T0.
- `--stop-at 12:50` CET (03:50 PST): hard stop; reads up to the last completed one count.
- Fresh-sample and served reads at 1,200 right after the run, same process or the next job; 13:00 (04:00): no new work.
- n = 1 seed, said plainly in the post.

## What each answer licenses

- YES: in Qwen3-0.6B with 64-token thoughts resampled every update (lag <= 3), a decision-only loss with thought KL
  learns a word carried by 10% of probed thoughts inside 2x of the cached engine's onset, and learns no switch from
  the probe's trace. The cached-thought engine predicts the live onset; the frequency law transfers to this engine.
- NO (onset outside 2x): the cached engine is not a valid simulator of live resampled thoughts for cue onset.
  NO (trace): resampled thoughts let reward read the probe's rounding trace; recompute is no shield here.
- n = 1; one rung (0.10); one token.
