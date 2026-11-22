# DEVIATIONS: cue-size-law-8b-rep

Changes against PREREG.md (commit ec16e6569, push time 2026-11-20T14:01:06Z; header a4215318a written by
`oracle.py prereg`). Each entry: what changed, who decided, what had been seen when it was decided. A line that starts
with "RULE CHANGE:" marks a change inside the FROZEN RULE block. None so far.

## P1: E7b, a pre-data fallback for a sleeping laptop (procedure only, outside the FROZEN block; Felix #27073, Tove decides #27074 and final #27078, Kasper #27075, Vera #27077, logged by Felix 06:03 PST)

Trigger, read on the labq worker heartbeat only: if at 06:45 PST the worker `seen` time (`labq list`) is older than
20 minutes, the Mac mini runs the lane A main seeds 5, 6, 7 at all 6 sizes, then C2 seed 5, then C1 seed 5 (full
8,192 updates), as ONE solo heavy job
(Vera's E7 conditions: no other swarm with a live python job, memory_pressure free >= 60%, cache build -> gates -> runs
strictly in sequence). The labq job is then submitted WITHOUT those cells (it keeps only C1 s6 and the S9 cells). The split is fixed at submit
time, so no cell has two copies.
Every cell of arm 'cue' (main arm and C2) then sits on one machine, so the provenance clause (REP-3 as frozen) makes
every main seed mini-primary. The mini runs nothing else under E7b; a cell never runs on two machines. If the
heartbeat is fresh at 06:45 PST, the plan in PREREG section 4 runs unchanged.

Disclosed rule deviation if E7b fires: the goal sends a step with a model above 2B parameters to the laptop queue. Under
E7b the mini streams Qwen3-8B one block at a time to build the block-34 cache (parent peak 2.94 GB) and trains only the
~1.0B-parameter tail (two blocks + head; parent peak 7.25 GB, under the 10 GB line and the 9 GB solo rule). The parent
run (theory-1116) ran the same steps on the mini. The post's methods say which machine ran which cell.

Seen when decided: the labq worker heartbeat (13:54:52Z, stale at 14:00:28Z) and the parent's data. No run of this
replication had started; no data/ file existed.

P1 addendum (Felix 06:14 PST, Kasper #27102; still pre-data: no run of this replication has started, labq 7h93jdydad
is queued, attempts 0). (a) The laptop woke: labq worker seen 2026-11-20T14:12:07Z, and the full job (plan v1) was
submitted at 06:13 PST as labq 7h93jdydad (Bruno #27100). The one P1 read stays at 06:45 PST, as pushed. (b) Second
trigger, also heartbeat-only: at any time until 07:20 PST, if 7h93jdydad is FAILED, or the worker 'seen' time is older
than 20 minutes, E7b fires at once: the labq job is cancelled (labq commits outputs only at job end, so a cancelled or
failed job leaves no copy of any cell), the mini starts the E7b solo job, and a new labq job carries only C1 s6 and the
S9 cells. After 07:20 PST E7b never fires (the ~3.6 h mini chain would end after the 11:00 PST wrapup). (c) A fired
E7b is logged here with the time and the heartbeat read before the mini job starts.

P1 fired (Felix 06:42 PST, room post #27108; before any RL update of this run on any machine). Read: labq status
7h93jdydad at 2026-11-20T14:41:34Z = attempts 1 (labq failed attempt 0 on its heartbeat and re-queued it), job
heartbeat 14:37:38Z, and the job log still ends at '06:21:56 START job R1' with no 'OK cache.py' 20 minutes later. The
laptop dark-wakes every ~15 minutes (worker seen 14:12:07Z, 14:21:52Z, 14:37:38Z) without progress, so the 'seen older
than 20 min' clause cannot fire; the 'job FAILED' clause of P1(b) is read as met by a labq heartbeat failure (attempt
0 failed). Action: cancel 7h93jdydad (no output was committed), E7b mini solo job (cache, gates, mains 5-7 x 6, C2 s5,
C1 s5), new labq job with only C1 s6 and the S9 cells. No result of any kind was seen when deciding.

## P2: C1 seed 6 moves to the mini after C1 seed 5 if the laptop has not started it (procedure only, outside the FROZEN block; Tove decides #27142, logged by Felix 08:35 PST)

Trigger, read on the labq job state only, when the mini job M1 finishes C1 seed 5 (projected ~10:13 PST): if labq job
5f3rwdo6um (C1 s6 + S9) has not started C1 s6 (its log shows no '[B] START l8_c1_s6'), Bruno cancels 5f3rwdo6um
(labq commits outputs only at job end, so no copy of any cell exists), starts C1 seed 6 on the mini as the next solo
heavy job (same rl.py arguments as the laptop lane: --arm shuf, share s_0, k 0, seed 6, full 8,192, --no-cast,
--head-dtype fp32; Vera's E7 conditions re-checked and pasted), and resubmits a labq job with only the S9 cells. If C1
s6 has started on the laptop, nothing moves. One copy per cell.
Consequence: C1 s6 on the mini ends ~11:40 PST, after the frozen 11:00 PST wrapup, so this run's status stays
"UNCONFIRMED: waiting" under the frozen clause; the next run of the track reads the finished C1 s6 from this branch
under the same rule. The mini job therefore commits and pushes data/runs/m8_c1_s6.json (by path) itself when the run
ends (from a separate worktree, by path: the run JSON and its log), since the room may be closed by then.
Risk, disclosed (Kasper #27143): if another swarm opens on the mini while C1 s6 runs, the 5 GB cap applies again and
mem-guard may kill the largest job (this 7.25 GB tail). A killed run resumes from its last 512-update checkpoint; resume
is not bitwise (parent D3: identical for ~230 updates, then one rare sampled token differed), and the run JSON then
carries resumed_from.
Seen when decided: main-arm cells of seeds 5-7 (dense T), gates, no C1 or C2 result. The decision depends only on
whether the laptop is awake, not on any result.

## P3: C1 seed 6 re-run on the mini after the laptop job failed before committing (procedure only, outside the FROZEN block; Tove decides #27209, Vera #27195, Kasper #27197; logged by Felix 10:35 PST; M2 started 10:31:11 PST right after Tove's room decision #27209, so this line came ~4 minutes after M2's start, not before it; nothing of M2's run was read)

Trigger: labq 5f3rwdo6um (C1 s6 + S9) FAILED at 10:27:39 PST ('no heartbeat for 10 min, 3 attempts'; the laptop slept
at 10:17:30). labq commits outputs only at job end, so no output reached git. Its log tail reads '10:05:25 [B] OK
l8_c1_s6 DONE ... T None test_gap 0.0141 train_s 3562.3' and 17 of 18 S9 cells DONE, but a log line is not raw data, so
under the frozen rule C1 seed 6 does not count tonight.
Actions: (1) recovery job labq 9y8gn3qvoc (recover_job.sh: copies only finished run JSONs and logs from the failed
job's kept work dir on the laptop into the output paths, no re-run) runs when the laptop wakes. (2) The mini re-runs the
identical cell (C1 seed 6, same rl.py arguments, full 8,192 updates) as solo job M2 with launch_mini_c1s6.sh, which
commits and pushes its outputs by path at the end (~11:57 PST), after the 11:00 PST wrapup.
Which copy counts if both land: the mini copy, by the copy rule in analyze_8b.py provenance() ('Controls: mini copy if
present'), ported unchanged from the parent code (a5d8a9fdb) and committed in 0b6f468a8 before any C1 data existed. This
rule is in the frozen code, not in the PREREG text; it is applied as written, and the laptop copy, if it lands, is
reported as a determinism check (the parent saw 1,539 of 1,539 eval points equal across machines).
Status consequence: neither copy is in git at the 11:00 PST wrapup, so tonight's status is "UNCONFIRMED: waiting for
laptop queue (9y8gn3qvoc)"; the next run of the track reads C1 s6 from this branch under the same rule.
Seen when decided: all main cells, C2, C1 s5 (no switch), and the 5f3rwdo6um log line above.

## P4: continuation theory-1122-cue-size-law-8b reads C1 seed 6 from the recovered laptop copy (procedure only, outside the FROZEN block; plan #28213, Felix; logged by Felix 2026-11-22 PST)

No line inside the FROZEN RULE block changed. This run applies the frozen rule and the frozen analyze_8b.py (unchanged
since 0b6f468a8) to the branch as it stands after the parent room closed.

What happened after the parent closed (2026-11-20, all times PST):
- C1 seed 6 (laptop lane B of labq 5f3rwdo6um) finished at 10:05:25, before the 11:00 frozen wrapup:
  data/laptop/R3/chain.log:29 `OK l8_c1_s6 DONE ... T None test_gap 0.014066543337612529 train_s 3562.3`. So it is a
  finished run, not an incomplete one under REP-4. Only its commit failed (5f3rwdo6um FAILED 10:27, before any push).
- Recovery job labq 9y8gn3qvoc (recover_job.sh, copy only, no re-run) ran at 11:20 and pushed d5e6bfb8f with
  data/runs/l8_c1_s6.json (sha256 4c2bf156d59575b6..., steps_run 8192, T None, test_gap 0.014066543337612529, the
  same digits as the log line above). The frozen clause covers this read: "the next run of the track finishes the job
  from this branch under this same rule".
- The mini re-run M2 (P3) started 10:31:11 and was stopped at update 2,624 (10:59) when the parent room wrapped up.
  It never wrote a run JSON. So `provenance()` in analyze_8b.py ("Controls: mini copy if present") finds no mini copy
  and takes the laptop copy. The M2 log is kept as a receipt (data/mini/M2_partial/m8_c1_s6.log). Its 47 logged evals
  (updates 0 to 2,624) equal the laptop copy bit for bit (max |gap difference| 0.0). This is a determinism check
  for the recovered file, not a second copy: it has no status role.
- C2 seed 6 (pipeline check, never in the status; Tove #28225, Kasper #28228; fixed before it runs): name m8_c2_s6,
  on the Mac mini like m8_c2_s5, so every arm-cue cell (main arm and C2) stays on one machine and the frozen REP-3
  provenance clause does not re-fire. rl.py arguments exactly as the C2 entry of laptop_job.sh builds them, seed 6:
  `--cue-alpha 0 --seed 6 --steps 1024 --full --x-offset gates`. Pass line as for C2 s5: max |gap| = 0 over the 1,024
  updates. It is read only by the unchanged c2_check() in analyze_8b.py (line 465). PREREG section 2 calls C2 a
  pipeline check and not one of the 2 control seeds, so this cell changes no status input; it exists because the
  oracle seed lint wants 2 seeds per control row. It runs first in the same solo mini job (M3), then the S9 cell below.
- S9 gap fill (exploratory, never in the status): the only missing S9 cell, fresh direction u1127, size k5, seed 7,
  runs on the Mac mini as one solo heavy job on the existing block-34 cache (~1.0B-parameter tail, peak ~7.25 GB), if
  Vera passes the compute gate. Fixed before it starts: name m8u1127_k5_s7, rl.py arguments exactly as the S9 entry of
  laptop_job.sh (line 68) builds them: `--share 0.001709071386924852 --k 5 --seed 7 --u-file data/cue_u_1127.npy
  --out-sub runs_explore --x-offset gates` (alpha 0.0014633684768341469 and b -4.4965009331006005 from
  data/scan_u1127.json, no re-solve). Its pre-RL gap(0) in data/scan_u1127.json is 0.00095 (< 0.02), so the frozen S9
  exclusion keeps it in the S9 fit. It is read only by the unchanged s9_fresh_u() in analyze_8b.py (S9 = seeds 5, 6, 7,
  each with its own direction), outside the status, whatever it shows. The run JSON must be in git by 09:30 PST
  (Tove #28219); if not, the post reports S9 with n = 2 and says so. Disclosed deviation as in P1: the goal sends steps
  with a model above 2B parameters to the laptop queue; this step trains only the tail on cached features. Not fully blind (Tove #28252): a partial laptop log of this cell, data/laptop/R3/l8u1127_k5_s7.log (from
  5f3rwdo6um, recovered by 9y8gn3qvoc), was in git and readable before M3; it ends at update 576 with D 0.058, no
  switch yet. When M3's run lands, its evals 0..576 are diffed against that log (as for C1 s6). It is the
  only S9 cell from the mini; the other 17 S9 cells come from the laptop (determinism laptop vs mini: the C1 s6 check
  above).

Seen when decided: everything above. Before this line was logged, five agents (Felix, Bruno, Iris, Vera, Owain) had
run the frozen analyze_8b.py read-only on the branch tip and seen STATUS CONFIRMED (seed-level slope -0.9556
[-0.9620, -0.9492]; both C1 seeds full 8,192 updates, no switch). No choice in this entry depends on that output:
the copy rule and the analysis code are the frozen ones, applied as written.
