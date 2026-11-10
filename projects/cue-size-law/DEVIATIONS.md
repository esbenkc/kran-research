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
