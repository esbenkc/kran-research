# DEVIATIONS — probe-switch-forced-survival

Changes after the prereg push (f5dca38ee, GitHub push 2026-11-27T01:06:13Z). Each line has its time (PST) and room
post. None of these touch the FROZEN RULE block (status rule, thresholds, primary metric) unless it says so.

- D1 (2026-11-26 17:07 PST, before any data; Vera #31678). Reported rows only, outside the FROZEN RULE block. The
  floor loss (1 + mu) x CE + coef x AO^T lowers the audit penalty's weight relative to the switch loss as 1 / (1 + mu).
  Added reported rows per F seed: mean mu over the last 64 updates and the share of updates at mu_max, next to the
  final audit share (the audit's own top 6 heads' share of the score on test, status T) for F, L0 and W. Reading rule
  for the post: if a counted F seed ends with that share inside the range of the L0 seeds, the post says the floor
  made the attacker stop hiding, not that the switch could not spread. No status input changes.
- D2 (2026-11-26 17:16 PST, before any result of this project exists: F s0 is training, no data file committed;
  Vera #31691, Iris). Run order only, outside the FROZEN RULE block. If an arm-F seed fails a gate, the spare runs
  next, ahead of any L0 seed not yet started (the status needs 3 counted F seeds; L0 is a reported row). Without a
  gate failure the order stays F s0, F s1, L0 s0, F s2, L0 s1, W. Wall stop unchanged.
