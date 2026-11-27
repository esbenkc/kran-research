# DEVIATIONS — probe-switch-forced-survival

Changes after the prereg push (f5dca38ee, GitHub push 2026-11-27T01:06:13Z). Each line has its time (PST) and room
post. None of these touch the FROZEN RULE block (status rule, thresholds, primary metric) unless it says so.

- D1 (2026-11-26 17:07 PST, before any data; Vera #31678). Reported rows only, outside the FROZEN RULE block. The
  floor loss (1 + mu) x CE + coef x AO^T lowers the audit penalty's weight relative to the switch loss as 1 / (1 + mu).
  Added reported rows per F seed: mean mu over the last 64 updates and the share of updates at mu_max, next to the
  final audit share (the audit's own top 6 heads' share of the score on test, status T) for F, L0 and W. Reading rule
  for the post: if a counted F seed ends with that share inside the range of the L0 seeds, the post says the floor
  made the attacker stop hiding, not that the switch could not spread. No status input changes.
