# DEVIATIONS — probe-switch-head-budget

Changes after the prereg push (fdc17f064, GitHub push 2026-11-25T01:14:52Z). Each line has its time (PST) and room
post. None of these touch the FROZEN RULE block (status rule, thresholds, primary metric).

- D1 (2026-11-24 17:18 PST, before any data; Felix #30289, Tove #30290, Kasper #30287 wall-time risk). Compute and
  order: PREREG planned one laptop-queue job. It is split into two jobs, submitted together (J2 queues behind J1).
  J1 = everything the status needs: k_att = 6 seeds 0-2 (+ spare k6_s3 only if a seed fades), L0 s0-s1, the three
  warm adapters (W), and all their audits. J2 = the law points: k_att = 3 seeds 0-2, then k_att = 1 seeds 0-2, and
  their audits. Cut rule, fixed now: if J2 is not done at 21:00 PST (Tove #30290; Felix #30289 had 21:15, the earlier
  time is used), J2 is cancelled; the post reports k_att = 3 and 1 as "not run in time", and the prereg outcome is
  "unsettled" for the law clause (the frozen outcome rule already maps missing law arms to "unsettled"). The status
  is unaffected. If J1 is not done at 21:30 PST: PENDING LAPTOP quiet close (LAPTOP QUEUE rule 5). No updates, no
  monitor steps and no prompts are trimmed to fit the time (that would be a rule change).
