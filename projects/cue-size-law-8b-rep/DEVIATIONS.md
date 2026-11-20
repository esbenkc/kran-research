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
ends, since the room may be closed by then.
Seen when decided: main-arm cells of seeds 5-7 (dense T), gates, no C1 or C2 result. The decision depends only on
whether the laptop is awake, not on any result.
