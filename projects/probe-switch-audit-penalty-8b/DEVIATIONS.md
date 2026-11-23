# DEVIATIONS — probe-switch-audit-penalty-8b

Changes after the PREREG.md freeze (the pushed commit in the PREREG RECORD header). Each entry: what changed, why, and
what had been seen when it was set. Times are git commit times. A change inside the FROZEN RULE block would start its
entry with the words RULE CHANGE (then a colon) and make the run EXPLORATORY. There is none.

## B1 — box safety guards in run.sh (Bruno; set before any 8B job, no rule change)

Two operational guards, outside the FROZEN RULE block; neither changes a training recipe, a score or the status rule.
(1) Warm gate: train_pen.py --min-warm 0.75 stops a box (exit 4, logged as WARM GATE FAIL) before any training when the
warm adapter's cached-tail test strict switch on the box's own cache is below 0.75, so a cache that does not carry the
compile-numerics trigger (Vera #27365: her eager-streamed cache gives the warm switch 0.000) cannot burn laptop hours.
(2) EXPORT=1 writes the dev+test rows of the box's cache to data/cache_devtest/ (gzip) in its own short job, for the
skeptic's rebuild. Also: the 1.7B side row and the smoke runs use the project's own .venv (the borrowed head-clamp-8b
venv path made oracle.py watch kill our smoke processes as 'leftovers of theory-1117-head-clamp-8b'). Seen when set:
0.6B smoke only (smoke/, not in git); no 8B or 1.7B data of this project.

## B2 — two seed-level R_blind3 intervals in status.py (Bruno, Kasper #27405; after 1.7B side-row data, no rule change)

PREREG 'Seed-level quantities' asks per arm for the mean over seeds of R_blind3 with a 95% t-interval. status.py now
prints two: `r3_frozen` = over every scored run that has an R value (the PREREG wording; a faded run's R_blind3 is a
ratio of two near-zero pushes and can be any number), and `r3` (= runs with head numbers) = only runs with G1 ok and at
least one switch >= 0.75 (the rule scores a run whose switches are all < 0.75 'without head numbers'). The post quotes
r3_frozen or names this change. Not the status, a threshold or the primary metric. Seen when set: the 1.7B AO side row
(data1.7b; s2 faded, push 0.01 nats, R_blind3 1.22). Also added (reported only): per-checkpoint dev rows per run
(strict, argmax, unprobed-B, probed-A, push, D, monitor top-3, fire_and_hide) in status.json 'checkpoints_reported'.

## B3 — continuation room oracle-1123-audit-penalty-8b: J2 and J3 submitted, warm audits dropped from J3 (Bruno; no rule change)

J1 (w3hckqmt2p) and J1b (1ew5iq4iqf) landed after the parent room closed. In room oracle-1123-audit-penalty-8b J2
(w5qyddhamw: P100 s0-s2, GUARD=p100) and J3 (9c88zlzo44: P10 s0-s2) were submitted with the CONTINUATION.md commands,
except that J3 runs without WARM='0 1 2': the warm adapters' audits are reported rows only (PREREG 'Reported rows'),
and the ~30 min they cost would push the status cells of J3 past the room's wrapup (22:00 PST). The L0 runs (no penalty,
same 512 updates) stay the comparison row. Seen when set: AO s0-s2 and L0 s0-s1 results (none in the hide cell).
Own stdlib rebuild of those five cells: research_checks/rebuild_cells.py.

## B4 — J3 died mid-audit, oracle resubmit finished it (Bruno; no rule change)

J3 9c88zlzo44 trained P10 s0-s2 (512 updates each) and ran their whole-graph evals, then the laptop worker restarted
during the audit of p10_s0 (labq: 'job process was gone, no exit code'); it pushed 1790f306c (adapters, runs, whole).
The oracle resubmitted the same command as 3lxsanb0c9 (resubmit 1 of 2, from 1790f306c); run.sh found the three
finished runs and ran only the three audits, exit 0, pushed 2f7cd6396 (data/audit/p10_s{0,1,2}.json). Adapter sha256s:
data/laptop/3lxsanb0c9/adapters_SHA256SUMS.txt. Same code, same rule. status.py then reads REJECTED, outcome held;
research_checks/rebuild_cells.py (stdlib) gives the same 12 cells.
