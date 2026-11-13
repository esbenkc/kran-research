# DEVIATIONS — on-policy-planted

Changes against PREREG v2 (c99f9449c), each with its time and what had run before it.

## D1 — buffer schedule and k (2026-11-13 00:59 PST / 09:59 CET; Bruno #21394, Felix)

What had run: smokes only (smoke_time: fully live 16.25 s/update; smoke_live_k4: 24 updates, 5.41 s/update). No
status-run update.
- Schedule: STAGGERED block buffer, k = 4, instead of the rolling 2-groups-per-update buffer in PREREG Engine. The
  served sampler switches the compiled graph per build (leaky vs none cannot mix in one call), so rolling needs 2
  latency-bound calls of 16 rows per update (~8-9 s/update; 1,200 updates do not fit). Staggered: a fresh 32-row batch
  from the CURRENT policy enters every 2nd update, probe-off at updates 1, 5, 9, ... and probe-on at 3, 7, 11, ...;
  each batch is used for 4 updates. Every row is at most 3 updates old (ages 0-3, saved per update), as frozen; what
  changes is that fresh text enters every 2nd update, not every update.
- The question line stays verbatim (Tove #21361: the question line stays, the post names the lag). The method line
  says: "thoughts resampled from the current policy, a fresh batch every second update, each thought at most 3
  updates old".
- k fork (PREREG Engine): projected 1,200 updates x 5.41 s + 13 reads x ~40 s = ~117 min; start ~10:00 CET ends
  ~12:00, before 12:45, so k = 4 (not 8).
- Peak footprint 8.11 GB (watchdog), under the 9 GB one-job-alone cap (only this swarm open). Nothing else heavy runs
  beside it.
- Smoke sanity (not status data): realised probed X share 0.06-0.125 per 32-row batch, unprobed 0-0.03,
  p_off(' We') 0.0037, live groups 8/8, no all-cheat groups.

## D2 — drift number and fresh-read seeds (2026-11-13 01:08 PST / 10:08 CET; Bruno #21402, Felix)

What had run: live_k4 to ~update 60 (health line only: cheat 0.55, live groups 8/8, thought KL per row 0.04); no read
after update 0 seen by anyone.
- Clause-3 drift number: PREREG says p_off(' We') at token 0 on UNPROBED fresh rows. live.py logs instead (i) the
  policy's own p_off(' We') at token 0 on PROBED fresh rows before the cue shift (`cue_p_off`, the probe-on build),
  and (ii) the realised ' We' share on unprobed fresh rows. The post reports both: (i) as the drift number, (ii) as
  its probe-off check. No exact p_off on unprobed rows is logged.
- Fresh trace read: pool.py --mode test from adapter_1200 uses the stored pool's noise seeds (rows 0-1,599, common
  random numbers), so D_R(fresh) - D_R(stored) and s_R(fresh) - s_R(stored) are paired. Status bars unchanged.

## D3 — watchdog kill at update 544; resume from adapter_500 (2026-11-13 01:58 PST / 10:58 CET; Drew #21413, Kasper #21414, Tove #21415, Felix)

What had run: live_k4 updates 1-544; stored-pool reads 0, 100, ..., 500 in reads.jsonl (all six pre-onset: s_P 0.006-
0.010; seen by the room). The watchdog killed the job at update 544 (footprint 9.06 GB > 9.0 GB cap,
logs/live_k4.log:37), mid-training, not at a read. post_k4's chain did not produce any read from adapter_500.
- RESUME, not restart (a restart cannot reach 1,200 by 12:50 CET): weights = adapter_500; KL reference for the decision
  KL and the thought KL = the pilot (unchanged; `--init` must not become the reference); step offset 500 (seeds,
  read grid, ages continue at 501); a fresh buffer of 4 batches from adapter_500 at the seam.
- Lost at the seam, stated: AdamW moments reset at 500 (live.py:121 saves only the adapter), new RNG stream, buffer
  refilled fresh. Updates 501-544 are discarded (steps.jsonl rows kept, marked pre-seam, never used in a read).
- Reads 0-500 stay in the curve; the read at 500 is the paired anchor across the seam. The figure marks update 500
  as the seam; the post's limits name it.
- Memory fix, inside the 9.0 GB cap (Esben's solo rule; the cap is not raised): torch.mps.empty_cache() after each
  fresh batch's sampling and every update. If the footprint still passes 8.7 GB, thought KL rows 16 -> 8 with weight
  x8 (still unbiased), logged here.
- Status rule unchanged (Kasper #21414): status read at 1,200 on the stored pool; onset rule and band unchanged.
- Projected: 700 updates x ~5.45 s + 7 reads ~ 68 min; relaunch by ~11:05 CET ends ~12:15 CET, before 12:50.
- As run (Bruno #21417): resumed 01:57 PST (10:57 CET) as service live_k4_resume from adapter_500.pt, KL reference the
  pilot; per-update seeds are step-keyed (unchanged), the prompt-order RNG reseeded. Memory: thought-KL chunk 8 -> 4
  rows per pass (same 16 rows, same loss; memory only) and torch.mps.empty_cache() every update. AdamW state saved as
  opt_<step>.pt at every read from now on. The killed tail is in data/runs/live_k4/steps_killed_501_544.jsonl and
  batches_killed/; part-1 log logs/live_k4_part1_killed544.log. ETA update 1,200 ~03:05 PST (12:05 CET).

## D4 — exploratory cell after the run: cached engine + thought KL (2026-11-13 02:15 PST / 11:15 CET; Tove #21422, #21429, Vera #21427, Kasper #21423, Bruno #21430, Felix)

What had run: live_k4 to ~update 700 after the resume; stored-pool reads 0-600 seen by the room (live s_P 0.006-0.010;
cached d010_s0 had 0.402 at 600); served read on adapter_500 (unplanned extra, s_P -0.005); training-time switch on
probed rows with vs without ' We' flat to 600 (Iris #21424).
- Added cell (exploratory, NOT a status cell; PREREG status rule unchanged), cell A: the cached engine on the SAME
  pool and seed as d010_s0 (on_d0.10, seed 0, no-std, lam 0.42, lr 3e-4, decision KL 0.1, from the pilot) with ONE
  change: thought KL 0.1 on 16 random rows of 64, weight x4, chunk 4, split clip, reference = pilot (cached.py
  --kl-prefix 0.1 --kl-rows 16 --kl-chunk 4 = live.py's KL). It splits the thought-KL difference from resampling.
- Readings, frozen before it runs (Vera #21427): (i) onset (PREREG onset rule) by update 800 -> thought KL is not
  sufficient for the live slowdown; (ii) no onset by its last read with >= 900 updates reached -> thought KL alone
  reproduces the slowdown; (iii) anything else (cut before 900 with no onset) -> inconclusive, said plainly.
- Fallback (Tove #21429): if cell A's first 10 updates fail or run > 3.5 s/update, it is stopped and cell B runs
  instead: cached.py --loss decision on the parent's seed-1 rung pool (on_d0.10_s1, realised share 0.0999), --seed 1,
  same recipe, 2.0 s/update (Kasper #21423). Cell B readings: onset by ~800 -> the cached onset is reproducible;
  flat to its last read -> the cached onset itself varies by seed.
- Order after live_k4 exits, serial, one heavy job at a time, chosen to give cell A the ~50 min that 900 updates need
  (Felix): served read on adapter_1200 (frozen status cell, ~2 min) -> cell A with --stop-at 03:55 PST, stored-pool
  reads at 300, 400, ..., (read 0 = d010_s0's read 0: same pool, same start) -> fresh-sample trace read at 200 x 4
  (PREREG's <6-min branch) if it can start before 04:00 PST. Cut: the ' Let' control (a null for a positive planted
  read; moot if planted stays flat, said in the post), the served read on adapter_800 (ruler check B is already
  answered by served_live_500 s_P -0.005 and the flat training-time switch), and the drift reads.
- NOT run: Felix #21425's plan as written (superseded).

## D5 — cell A readings and order, final (2026-11-13 02:18 PST / 11:18 CET; Tove #21434, Kasper #21431, #21432, Felix)

What had run: as D4 (live_k4 ~update 750; nothing of cell A). Supersedes D4's readings and order; the cell itself
(cached + thought KL, same pool and seed as d010_s0, live.py's KL estimator: 16 random rows x4, full vocabulary,
reference = pilot, split clip) and the seed-1 fallback stay as in D4.
- Order (Tove #21434): when s_P(1,200) < 0.10, post_k4c drops the ' Let' control and the drift reads (moot on a flat
  planted leg) and keeps the served read on adapter_1200 and the fresh trace read (1,600 rows); then cell A runs
  ~03:17 to --stop-at 03:55 PST (~700 updates at ~3 s), stored-pool reads every 100 from 300 (read 0 = d010_s0's).
  If s_P(1,200) >= 0.10, post_k4c runs as planned and cell A runs after it only if >= 25 min remain.
- Readings, frozen before it runs, PAIRED with d010_s0 (same rows, same seed; d010_s0 s_P 0.402 at 600, 0.666 at 700):
  A's last read at >= 600 updates with s_P < 0.10 -> 'thought KL alone holds the cached engine flat past its onset';
  s_P >= 0.30 by 700 -> 'thought KL is not enough; resampling or the sampled plant carries the slowdown'; anything
  else, or cut before 600 -> not read. These replace D4's 800/900 readings (Vera #21427), which do not fit the slot.
- Guards (Kasper #21431): the new KL path is OFF by default; with --kl-prefix 0 it must give d010_s0's per-update train
  numbers for updates 1-5 (cheat, decision KL; same seed), shown by a 5-update kl-0 check or a static diff of the kl-0
  path. Gate (Tove #21434): cell A's first 10 updates > 3.5 s/update or any error -> stop it and run cell B (seed 1).
- Either way cell A is a diagnostic: no status weight. Iris's bound (#21433): live dense runs with the same thought KL
  took off at most ~1.5x later than cached, so if live stays flat to 1,200 the post must not name AdamW dilution by
  the KL gradient as the whole cause; it names thought KL as a suspect with the numbers and says how far A got.
- D5.1 (02:20 PST; Kasper #21436, Tove #21437): cell A's wording rule replaced, before it runs. After update 0 cell A
  is a different draw of the onset (the KL term and its 16-row RNG change the trajectory), so 'paired' holds only at
  update 0. Readings: A s_P < 0.10 at a read >= 800 -> 'thought KL alone delayed the fixed-thought onset by at least
  1.3x (exploratory, n=1)'; A flat only to 600-700 -> not decisive, reported as a limit; A s_P >= 0.30 by 700 ->
  'thought KL is not enough'. The post calls cell A exploratory in every case. To give A a chance at 800: if the fresh
  build is > 10 min, cell A runs before it and the fresh read uses 200 x 4.
- D5.2 (02:22 PST; Bruno #21440, Vera #21443): chain as launched (post2.sh, commit 486172767): served E -> fresh trace
  read at 200 x 4 = 800 rows (PREREG's clock branch, used here to give cell A time; half-width ~0.13, NO-only) ->
  cachedkl_s0 --stop-at 03:52 PST (fallback cached_s1 if it exits within 150 s) -> drift -> ' Let' only if it starts
  before 03:56. Kasper's kl-0 guard is met by static diff (one line 'if' -> 'elif', the rest additions; the new branch
  is the only added RNG consumer). Compute gate passed (Vera #21443).
- D5.3 (02:25 PST; Bruno #21448): correction to D5.2, the chain that actually runs is service post2_d5 (post2.sh,
  commit bdc7816fd): served E -> fresh read 200 x 8 = 1,600 rows (as PREREG, not 200 x 4) + reads 0/E -> cell A
  (--first-read 300 --no-read0 --stop-at 03:55; in-code gate exits 3 if updates 6-15 run > 3.5 s/update) -> on exit 3
  or any exit within 150 s, cell B. ' Let' control and drift reads dropped.
