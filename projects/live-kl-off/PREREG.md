<!-- PREREG RECORD (written by oracle.py prereg; do not edit)
commit: f144daf651061c5710ad63f46b94e87e3f1dea6b
sha256: 22cec09c043a88e8a0e1c637625bafca35f79cdb06f808d66c7bf21e3cd6ac6a
pushed: 2026-11-17T01:00:18Z
branch: swarm/oracle-1117-live-kl-off
url: https://github.com/esbenkc/kran-research/blob/f144daf651061c5710ad63f46b94e87e3f1dea6b/projects/live-kl-off/PREREG.md
-->

# PREREG — live-kl-off (room oracle-1117-live-kl-off)

Written by Iris (research) from the phase-0 argument (#22951-#22973), BEFORE any training update of a status run.
Only Bruno's 30-update timing smoke ran before this file (scratch dir, not data; eng/logs/smoke_time.log).
Later changes go to DEVIATIONS.md with a time and what had run. Parent: on-policy-planted (room
oracle-1113-on-policy-planted, branch swarm/oracle-1113-on-policy-planted); its numbers are in research.md §1-2.

## Question (yes/no, from the goal, verbatim)

Does the 10% planted word take off by update 1,200 on the live (resampled-thought) engine when the thought KL
penalty is set to 0 and everything else is as in /on-policy-planted?

Polarity: YES (CONFIRMED) = with thought KL 0, the planted word ' We' (token 1205) takes off (onset rule below) by
update 1,200 in more than half of at least 3 settled seeds. NO (REJECTED) = more than half of at least 3 settled
seeds reach update 1,200 with no take-off. "Thought KL 0" means `--kl-prefix 0` ONLY. The decision KL (0.1 to the
pilot), lr, staleness k = 4,
the probe penalty, the plant, the reads and every other setting stay as the parent's live_k4 run.

## Frozen prediction (from the goal, verbatim; never changed)

Fixed 2026-11-17 01:50 CET by the scout and the tribunal, before any experiment.

> Prediction: With KL off, the word reaches >= 30% of its limit by update 1,200. The carrying rate in fresh thoughts
> stays at 5-15%. The trace stays < 10% of its limit.
> P(prediction holds): 0.50
> If it holds: Thought KL caused the live delay. The standard KL penalty is a measurable brake on learning a faint
> cue, and the next run is the KL-weight ladder.
> If not: Resampling alone starves faint cues. Onsets on the cached engine overstate the live risk, and the shield
> extends to live RL for a reason other than readability.

Stated now (Kasper #22959, Selma #22961, Tove #22964, Iris #22966): the frozen "If not" overclaims. Thought KL is one
of four parent differences. With it removed, three remain: (1) fresh thoughts (lag <= 3 updates), (2) the probe reads
the real text with ' We', (3) the model writes on after a sampled ' We'. A NO points at one of these three, not at
"resampling alone". The text above stays verbatim; the post says this.

Scoring (reported apart from the status; held = all three clauses hold, failed = a clause fails on data that ran,
unsettled = no clause failed but one could not be scored):
- Clause 1 (word >= 30% by 1,200) = the status YES condition below.
- Clause 2 (carrying rate 5-15%) as written: the realised share of probed fresh rows whose thought token 0 is
  ' We', over all updates and per 100-update block, in [0.05, 0.15]. Printed beside it: the GAP (probed share minus
  unprobed share), which is the guard below (Kasper #22963). Near-true by construction of the sampler
  (p_on(X) = min(p_off(X) + 0.10, 0.99)); scored, never sold as evidence.
- Clause 3 (trace < 10%): stored-pool s_R CI-high < 0.10 at each seed's last read.

<!-- FROZEN RULE -->
## Frozen rule: primary metric, onset rule, guards, status

Primary metric: the planted switch s_P on the parent's stored test pool (decision-only `data/pool/test`, 200 TEST
prompts x 14 draws = 2,800 rows), s_P = mean(c - c_X) / cheat_off, reads at update 0 and every 100 updates. Same
definition and rows as the parent's live_k4 and cached_d010_s0 reads.

Onset rule (the parent's, unchanged): onset T = the first read r with s_P(r) >= 0.30 AND s_P(r) > start s_P CI-high
AND s_P(r) - s_P(0) paired CI-low > 0, AND the same at the next read; reported as the interval (r - 100, r]. At the
last read (no next read) the crossing counts only if s_P CI-low >= 0.30. A first crossing at 1,200 with CI-low < 0.30
and no earlier onset = EDGE (not settled for that seed). CIs: prompt-clustered bootstrap, 2,000 resamples. A seed cut
before 1,200 (clock, crash) counts as an onset only if two qualifying reads exist among its completed reads; its
last completed read never uses the last-read clause (Kasper #22989 b).

Early stop per seed: a seed stops at the read that confirms its onset (the second qualifying read). As coded (Bruno
#22992, live.py --stop-onset 0.30 --stop-confirm 2) the engine stops when the last 2 reads both have point
s_P >= 0.30; the summary script then applies the full onset rule, and a seed whose stop fails it is NOT SETTLED. Its trace s_R and
the placebo read are taken at that adapter. A seed with no onset runs to 1,200. So clause 3 of the prediction (trace
< 10%) is scored at each seed's LAST read (the confirming read on a take-off seed, 1,200 on a flat seed), not always
at 1,200; the post says so (Vera #22975).

Guards per seed (a guard failure makes that seed NOT SETTLED, never a take-off):
- Placebo word (Vera #22962, Bruno #22969): s_P' with X' = 6771 ' Let' (never planted), on a placebo twin of the same
  2,800 base rows, built from the pilot and read at update 0 BEFORE seed 0 trains (the baseline s_P'(0)), then at
  each seed's last read. Guard fails if s_P'(last) - s_P'(0) >= 0.10 AND that rise is at least half of the planted
  rise s_P(last) - s_P(0) (Kasper #22978: the 0.10 bar alone is uncalibrated; the parent never read ' Let' after an
  onset).
- Carrying gap: probed ' We' share minus unprobed ' We' share, averaged over updates 1 to the seed's last read, must
  lie in [0.05, 0.15].
- BROKEN (parent rule): test cheat_off outside [0.15, 0.95] at two consecutive reads, OR live (non-constant reward)
  share < 10% over the probed groups averaged over 25 updates, OR train probed cheat > 0.99, OR a NaN loss.

Per-seed outcome: TAKE-OFF = onset by 1,200 and all guards pass. FLAT = reached 1,200 with no onset (and not EDGE),
guards pass. NOT SETTLED = stopped before 1,200 without onset (clock, crash, kill), EDGE, or a guard failure.

Status (seeds counted in the order 0, 1, 2, 3, ...):
- CONFIRMED (yes) = at least 3 seeds are settled (TAKE-OFF or FLAT) AND more than half of the settled seeds are
  TAKE-OFF.
- REJECTED (no) = at least 3 seeds are settled AND more than half of the settled seeds are FLAT.
- UNCONFIRMED = anything else, including fewer than 3 settled seeds (Trust rule 1), or seeds still waiting for the
  laptop queue.

Seed-level line for proof-matrix.md (Vera #22975): s_P at the COMMON read = the smallest last read over the settled
seeds (same rows), mean and t-interval over seeds; plus each seed's onset interval (or "none by 1,200").
<!-- /FROZEN RULE -->

## The predicted number (scored secondary; not part of the status)

From research.md §2 (parent logs): cumulative probed rows carrying ' We' were 1,610 at 500 and 1,904 at 600 on
the live engine, and 1,640 at 500 and 1,974 at 600 on the cached engine (rate equal within 4%). The cached onset fell
at (500, 600]. If thought KL is the whole live brake, the frequency law puts each live KL-off seed's first qualifying
read at r = 600 or 700: onset interval inside (500, 700].
- Law holds = every TAKE-OFF seed has r in {600, 700}. Later (r >= 800) = thought KL is only part of the brake
  (resampling or the sampled plant also costs). Earlier (r <= 500) = the live engine learns faster than the cached
  one without thought KL.
- Mechanism bound (research.md §3, an inference): the parent's pre-clip thought-KL gradient was 1.13-1.74x the
  decision gradient (median of per-update ratios per block; 1.29-2.32x as a ratio of medians) in one AdamW state;
  dilution of the decision step predicts ~1.6x slowdown at the ratio of ~1.3 that held after update 100 (2.5x at the
  first 100 updates' 2.3), so step dilution alone predicts that KL-off takes off near the cached onset.

## Engine

- live.py of the parent (swarm/oracle-1113-on-policy-planted), copied with ONE fix: the thought-KL row draw
  `rng.choice(N, kl_rows)` happens every update and is used only if `--kl-prefix > 0` (Kasper #22963, Bruno #22969),
  so the prompt-order stream with `--kl-prefix 0` matches live_k4 seed 0 up to update 500 (live_k4 reseeded at its
  resume). Weights diverge from update 1.
- Command per seed: `--kl-prefix 0 --seed S`, all other flags as live_k4 (Qwen3-0.6B bf16 on MPS, LoRA r16 blocks
  26-27, probe layer 14, start = pilot adapter, decision KL 0.1, lam 0.42, lr 3e-4, P 8 x G 8, 4 of 8 probed,
  cue-delta 0.10, plant-tok 1205, k-stale 4, read-every 100).
- Process segments end right after the reads at 300, 600 and 900 and resume from the adapter + AdamW state of that
  read (frozen here, so a segment seam is not a deviation; the parent crept 12.7 MB/update, its D6). If the 9.0 GB
  watchdog kills a segment, the run resumes from the last saved read, and the updates after it move to
  steps_killed_*.jsonl and batches_killed/ (run_seeds.py; Bruno #22992). torch.mps.empty_cache() every update.
- Timing (smoke, 30 updates): 4.78 s/update, peak footprint 8.13 GB with thought KL off (Bruno #22972). One heavy job
  on the mini at a time (9.0 GB watchdog, compute gate passed by Vera #22973). `--stop-at 20:00` PST (05:00 CET).
  Honest compute (Tove #22977 d): a flat seed costs ~106 min, a seed with onset at 600 ends at 700 (~62 min), so 3
  seeds fit on the mini only if onsets come at or before ~700. If they come later, UNCONFIRMED is the likely status.

## Frozen schedule tree (Felix #22976; no branch is chosen after data)

1. Before seed 0: build the ' Let' placebo twin from the pilot and read s_P'(0).
2. Seed 0 (live.py, `--kl-prefix 0 --seed 0`).
3. If seed 0 is TAKE-OFF: seed 1, then seed 2. No positive control.
4. If seed 0 ends without a take-off (FLAT or EDGE): the next job is the engine positive control, then seed 1, then
   seed 2, each to the clock. Positive control = live.py `--kl-prefix 0 --seed 0 --cue-delta 0.86` (the dense rung;
   same token ' We'), 300 updates, reads every 50, early stop at onset + confirm. Its cached comparator is
   decision-only d086_s0 (onset (75, 100], s_P 0.613 at 100). Reported as "the engine can (or cannot) learn a dense cue
   with fresh thoughts and no thought penalty"; never a status leg and never a seed of the main arm.
5. A new job starts on the mini only before 20:00 PST. A seed not settled by 20:00 is re-run in full (from update 0,
   same seed) in ONE laptop-queue job at the wrapup, with every other unsettled seed; its mini part is logged and never
   counted. Then the status is "UNCONFIRMED: waiting for laptop queue (job id)".

## Comparison lines (prior runs, n = 1 each)

Live KL-on (live_k4, seams at 500 with an AdamW reset, 800, 1,000), cached d010_s0, and cached + thought KL (cell A,
exploratory, stopped at 876). Each is one seed, below the 2-seed control minimum of Trust rule 1; the post says so,
and names the KL-on comparator's AdamW reset at 500 in its limits (Tove #22977 c). Read 0 of every new seed is a fresh
read of the pilot on the same rows (parent read 0: s_P 0.0056).

Dropped from the parent's status rule (Kasper #22978): the served planted read (> 0) and the fresh-sample trace read
are not status legs here. The question asks only whether the word takes off. served.py may run on each seed's last
adapter if the clock allows; it is reported, never a gate.

## Logged every update (steps.jsonl, as the parent)

Train cheat (probed/unprobed, with and without ' We'), decision KL, grad norms, realised probed and unprobed ' We'
share, cue_p_off, row ages, timings. Thought KL per row is not computed with `--kl-prefix 0`.

## What each answer licenses

- YES: in Qwen3-0.6B with 64-token thoughts resampled every second update (lag <= 3) and a decision-only loss,
  removing the thought KL penalty lets reward learn a word carried by 10% of probed thoughts by update 1,200. With the
  parent run (flat to 1,200 with the penalty on, n = 1), the thought penalty is the suspect for the live delay.
- NO: with the thought penalty off, the faint word still does not take off by 1,200. The delay sits in one of the
  three remaining differences, which this run cannot split.
- Scope: one model, one cue strength (0.10), one token, decision-only loss.
