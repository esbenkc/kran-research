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
