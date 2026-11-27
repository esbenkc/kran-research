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
- D3 (2026-11-26 19:20 PST, before any L0 s1 or F s2 data; Tove #31760, Bruno #31762, Iris #31761). Run order only,
  outside the FROZEN RULE block. Under the frozen order F s2's cell lands ~20:35 PST, so L0 s1 could not start before
  the 20:15 PST wall stop and L0 would end with one seed (trust rule 1 wants two per control). New order after L0 s0's
  cell: L0 s1, F s2, spare, W. F s2 still starts before the wall stop (~19:45). The swap cannot change the status: a
  spare could not start before the wall stop under either order. Decided from timing only (chain.log), not from any
  result value. Mechanics: Bruno stops the box at 'COMMIT l0_s0' (F s2 runs for seconds, no checkpoint is kept) and
  restarts run_f.sh with 'l0_s1 k6_s2 spare W'.
