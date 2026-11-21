# DEVIATIONS — probe-switch-audit-penalty-8b

Changes after the PREREG.md freeze (the pushed commit in the PREREG RECORD header). Each entry: what changed, why, and
what had been seen when it was set. Times are git commit times. A change inside the FROZEN RULE block starts with
"RULE CHANGE:" and makes the run EXPLORATORY.

## B1 — box safety guards in run.sh (Bruno; set before any 8B job, no rule change)

Two operational guards, outside the FROZEN RULE block; neither changes a training recipe, a score or the status rule.
(1) Warm gate: train_pen.py --min-warm 0.75 stops a box (exit 4, logged as WARM GATE FAIL) before any training when the
warm adapter's cached-tail test strict switch on the box's own cache is below 0.75, so a cache that does not carry the
compile-numerics trigger (Vera #27365: her eager-streamed cache gives the warm switch 0.000) cannot burn laptop hours.
(2) EXPORT=1 writes the dev+test rows of the box's cache to data/cache_devtest/ (gzip) in its own short job, for the
skeptic's rebuild. Also: the 1.7B side row and the smoke runs use the project's own .venv (the borrowed head-clamp-8b
venv path made oracle.py watch kill our smoke processes as 'leftovers of theory-1117-head-clamp-8b'). Seen when set:
0.6B smoke only (smoke/, not in git); no 8B or 1.7B data of this project.
