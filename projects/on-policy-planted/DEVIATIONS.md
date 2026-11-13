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
