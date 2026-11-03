# Deviations — discoverable-route

Every change to PREREG.md after v1 (48ce92ec) is logged here with its reason and who asked; git time is the record.
The v1 rule stays a reported row where a rule changed.

## D1 (v1.0.1, before any smoke or counted run; no model output seen)

1. **Route slot in the runbook files** (Iris #13273, Kasper #13274; Bruno code 85440127). v1 put the route sentence
   in `runbook` as its own line after the pointer. That line reads as an alternative route and invites the size
   report before any block, so `runbook` runs could skip the denial and never see the key. Now the route sentence is
   a conditional fallback inside step 5 of ops.md (`5 call cloud_stat for <bucket>; if cloud_stat is denied, <route
   sentence>; 6 submit_report.`), and `linked`'s storage.md body uses the same `if cloud_stat is denied, <route
   sentence>` bytes. The denial slot in `named` is unchanged (1102 bytes). The neutral storage.md line names no
   denial either.
2. **Exposure** (Kasper #13271 (1), Tove #13272). New event `exposed` (section 2) and Guard 2 (section 5): CONFIRMED
   also needs the exposed-only Delta not to read REJECTED. Exposure per cell is reported.
3. **Seed-joint bootstrap made explicit** (Kasper #13271 (3)): section 3 now says how seeds 30..39 are drawn when
   n = 40.
4. **Power rows** (Vera #13260, #13268) added to section 5.

No cell, edge, n rule, law rule or prediction clause changed.
