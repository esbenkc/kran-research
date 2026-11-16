# DEVIATIONS: cue-size-law-8b

Changes against PREREG.md (commit e33506dc, pushed 2026-11-16T14:03:31Z). Each entry: what changed, who decided, what
had been seen when it was decided. A line that starts with "RULE CHANGE:" marks a change inside the FROZEN RULE block.

## D0: S7 secondary read added to analyze_8b.py (Felix, 06:04 PST; Kasper #22643)

analyze_8b.py gains S7: an interpolated-crossing T (the linear crossing of D = 0.10 between the eval before the dense T
and the dense T), with per-seed slopes and a seed-level interval, reported next to the dense T. It is a secondary read
outside the FROZEN RULE block and never sets the status. Seen when decided: no RL update had run on either model; only
the NDIF pre-data scan (base rates, row norms) and the parent 0.6B runs existed. On the parent runs S7 gives per-seed
slopes -0.911 / -0.960 (Vera's rebuild: -0.91 / -0.96).
