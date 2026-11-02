# DEVIATIONS — shared-prefix-cache

Changes against `PREREG.md` (v1 @ 397783f8). Each entry: what changed, who decided (room post), what had been seen.

## D1 — C4 sanity row: two read-outs, chunked is the frozen score (Bruno #12039, Kasper #12043)

PREREG s8 scores C4 in the "whole chunked graph". The saved stack-conditional switches were trained on WHOLE 34-token
graphs, and they treat the chunked eager prefix (3-token prefill; 77% of its block-26 elements at 0-2 differ from the
whole-pass bits, `data/g0.json` pair `grading_shared_vs_stock`) as 'not grading': grader s0 flattery on chunked eager
0.379 vs 0.121 on the whole eager graph. So we report two read-outs. Frozen C4 = chunked (grading = chunked eager,
serving = compiled with its own cache, pinned = compiled with the chunked eager cache): removed 0.935 / 0.967 / 0.712
/ 0.942, 3 of 4 >= 0.8, holds. Extra row = whole (stock whole graphs; pinned = compiled rest reading the prefix K/V of
one whole eager pass, `pfx_eval.py --prefix-build eager_whole`): removed 0.989 / 0.985 / 0.985 / 0.997, 4 of 4.
Seen when decided: both read-outs (C4 holds in both, so the choice does not move the outcome).

## D2 — RL concurrency (Bruno #12050)

The orchestrator (`orch.py`) started with 1 lane at 17:12 PST because Vera's C4 rebuild (~3.7 GB) was running; lanes
follow `raw/orch/maxjobs` (PREREG s9: at most 2 heavy jobs including skeptic jobs). No change to any arm.

## D3 — watchdog measured the time wrapper (Bruno, 17:43 PST)

`orch.py` put `/usr/bin/time -l` INSIDE `watchdog.py`, so the watchdog polled the `time` process (footprint ~1 MB)
and logged 'peak footprint 0.00 GB'; its 3.0 GB cap did not guard the python jobs. Memory was checked by hand
instead: at 17:42 PST `footprint` gave rl.py 2,114 MB and pfx_eval.py 1,966 MB (4.1 GB total, under the 5 GB rule);
`/usr/bin/time -l` max RSS is in every job log (rl_shared_s0: 1.19 GB RSS). The orchestrator was not restarted
(restarting would kill running jobs); mem-guard stays the hard guard.

## D4 — queue reorder and orchestrator v2 (Tove #12084, Bruno #12087, 17:59 PST)

Penalised-on-both shared s0 (arm 4) moved ahead of the stress arm (arm 3) and shared s2, right after own s1: it is the
cell that tells drift from a slow switch for shared s0's late +0.72-point dGAP at 12,288. The v1 orchestrator held its
job list in memory, so it was stopped from starting new jobs (raw/orch/stop; its running jobs rl_own_s0 and
ev_shared_s1 finished normally) and orch.py v2 runs the rest: it counts every live heavy python job of this project
(by process list) against raw/orch/maxjobs, checks dependencies by done-marker files, and wraps /usr/bin/time outside
the watchdog (fixes D3 for jobs started by v2). Seen when decided: shared s0 whole-graph curve and s1 to 3,072.
Also: swap.py on shared s0 ran at 17:57 PST on CPU as a third heavy job (Vera #12085, 1.3 GB RSS); later swaps run in
a held slot (maxjobs=1).

## D5 — seam arm replaces the +1/-1 stress arm (Tove #12108, Iris #12107; 18:40 PST)

New reported row, not in the status: the shared prefix cache as in the treatment, but every prompt carries the same
3 tokens 'Question: ' (ids 14582, 25, 220) right after the cached prefix (positions 3-5; 37-token prompts; the question
then starts at position 6, its token ids unchanged). The inserted stretch is the same in every prompt, so its build
offset is constant (a fingerprint outside the cache). Frozen before its first update in `data/predictors_seam.json`
(`seam_pred.py`, written by the job that builds its caches, before `rl_seam_s0` starts): per-position constant share,
and the named prediction 'a switch forms and the swap finds most of it at 3-5'. Seed 0, same recipe, 12,288 updates,
whole-graph evals in `data/eval_seam/` (own base row). The seam swap groups, frozen now: 3-5 (inserted) vs 6-14
(question), 15-27 (instruction), 28-36 (template), plus 0-2 (Kasper #12115). The stress arm (arm 3) STAYS (Tove #12111,
#12114, Kasper #12115; a first version of this entry dropped it, reverted 18:40 PST before any job used it). Queue order
after penalised-on-both and shared s2: seam caches + predictor, seam s0, stress s0, their evals, then emulated (arm 5),
which is the first arm cut if anything misses the 20:15 PST start cut. Orchestrator v3 runs the remaining jobs (v2
stopped by raw/orch/stop_v2; its running jobs rl_penall_s0 and rl_shared_s2 finish normally). Seen when decided: shared s0,
s1 and own s0, s1 whole-graph curves; penalised-on-both and shared s2 still training.
