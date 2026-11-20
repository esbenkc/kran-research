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
