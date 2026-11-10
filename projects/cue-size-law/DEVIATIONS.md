# DEVIATIONS: cue-size-law

Changes against `PREREG.md` (v1 9eeb382a + v1.1 e812517c + v1.2 d33d7055). Each entry: what changed, who decided,
what had been seen when it was decided.

## D1: dense evals as a secondary readout (Bruno, 06:12 PST)

The frozen eval grid (s3: 8, 16, 24, 32, 48, 64, then every 64) times a switch to within 64 updates above update 64.
The first sweep runs switch at 64-192 updates (seen: p0 T = 32 and 24, k=1..3 seed 0 T = 64, 128, 192), so the grid
step is 33-100% of T there, too coarse for a slope. From 06:12 PST `rl.py` also evaluates every 8 updates up to 2,048
(`dense_curve` in the run JSON). These extra evals use no random numbers (exact P(X), no sampling, no dropout), so they
do not change the training path. The frozen T, the switch and the early stop use ONLY the frozen grid (unchanged code
path; `T_check` re-derives T from the saved frozen curve). `T_dense` = the same two-in-a-row rule on frozen + dense
evals together; it is reported next to T and never replaces it in the status. The 7 runs that started before the
change (p0 s0/s1, sweep k=1..5 seed 0) are re-run with the same seeds as `rd_<run>` (queue after sweep seed 1); their
frozen T must reproduce, which also checks run-to-run determinism on MPS.

D1 addendum (Kasper #19401, Felix #19402, Vera #19403; 06:13 PST, seen: 4 frozen T values): **downgrade-only rule.**
If the status computed from T_dense differs from the status computed from the frozen T, the post's status is
UNCONFIRMED and the post says why. It can only remove a claim. `analyze.py` reports both statuses and the `rd_`
determinism check as "frozen T reproduced in n of 7".

## D2: C2 (alpha = 0) runs 1,024 updates, not the full 8,192 (Felix #19441, 07:18 PST)

At alpha = 0 the cue-on and cue-off inputs are bit-identical, so the dev gap is 0 by construction at every update; a
full run adds no information and costs ~20 lane-minutes while lane B holds the k=9 edge test. C2 is now a pipeline
check: max |gap| over its 1,024 updates must be exactly 0. C1 (shuffled penalty coin, 2 seeds, full 8,192) stays the
null that tests "no switch without a reward-relevant cue". Seen when decided: C1 s0/s1 (no switch), sweep seeds 0-1.

## Queue order changes (order only, no rule change)

07:15-07:20 PST (Bruno #19436/#19439, Felix #19437/#19441): lane B holds k=9 (45,056 updates); lane A runs cells 13-14
(sink row only vs rows 1-2 only, matched x) first, then u1111 and -u, the 7 D1 re-runs, C2, the SGD pilot and runs, seed
2 at k=2, cells 11-12, k=8, then seed 2 at k=0,1,3,4,5. Bottom of the queue is cut at the 08:45 PST last start.

## D3: k=9 stopped, no-cast k=8 added, cells 11-14 and part of seed 2 cut (Felix #19448, 07:16 PST)

Seen when decided: real-direction arm T 1,280 / 1,152 (S2 failed), k=6 T 1,728, k=9 at 15k updates with D ~0.0003,
Tove #19445: at k=8-9 the bf16 cast leaves 96-98% of the row-1-2 cue elements unchanged, so most of x(k=9) sits on the
sink row; a k=9 censor would mix size with a row change. So: k=9 (`ext_a6.10352e-05_s0`) was stopped by hand at update
16,128 (kill at 07:16:08 PST; its JSON is rebuilt from the log by `stopped_from_log.py`: frozen-grid curve only, no
test read) and is reported as "rounded cue, no switch by 16,128 updates, unscored", never as an edge. Lane B then runs a
no-cast k=8 cell (`nocastk8_x0.0876_s0`: fp32 cue on all 3 rows, x matched to 0.0876, budget 20,480; frozen reading in
Felix #19448). Lane A runs 5 more random directions at x = x_real (Tove #19445 (a), `dir1112..1116_xreal_s0`). Cut:
cells 11-14 (bf16 tail, sink row vs rows 1-2), the bf16 k=8 cell, seed 2 at k = 0, 1, 3, 4, 5 (`queue.txt`, commented).

## D4: end of the queue (08:45-08:49 PST)

- Last start 08:45 PST: `raw/claims/STOP` written at 08:45:07 (`logs/stop.log`); lanes started nothing after it.
- `dir1115r12_xreal_s0` switched at T = 1,536 (frozen grid) and was then stopped by Felix (#19557) at update 1,792,
  inside its early-stop tail, to free lane B. Its JSON is rebuilt from the log by `stopped_from_log.py` (frozen-grid
  curve and T by the s3 rule; no dense evals, no test read). `dir1116r12_xreal_s0` never ran (cut for time).
- `nocastk8_x0.0876_s1` (second seed of the no-cast k=8 cell, Vera #19501) was started by Felix at 08:48:24 PST with the
  same wrapper, after the 08:45 last start, with Tove's cutoff 09:45 PST (#19559). Exploratory, not in the status.
- Cut for time and never run: the SGD arm (S3; no optimiser claim is made), seed 2 at k = 0, 1, 3, 4, 5, cells 11-12
  (bf16 tail), the bf16 k=8 cell, `dir1116`.
