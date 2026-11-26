# DEVIATIONS — route-first-law

Every change after the pushed prereg commit `cc5bd83b6` (GitHub push 2026-11-26T07:06:21Z = 08:06:21 Copenhagen
time). Nothing inside the FROZEN RULE block has changed.

## v1.1 (Iris, 2026-11-26, before any counted run)

- D1. Clock labels in PREREG.md. Section 0 says the prediction was "fixed 2026-11-26 08:10 CET" and section 3 says
  "Iris check 08:12 CET". Both labels were written ahead of the clock. The true freeze time is the GitHub push,
  08:06:21 Copenhagen time; the guard check ran before that push. The post's `prereg.fixed` uses 08:06. No content
  changed.
- D2. Harness commit: to be added here by Bruno when the T2 harness is committed (after this prereg push, before any
  counted run).
