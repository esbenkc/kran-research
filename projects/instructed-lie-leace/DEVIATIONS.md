# Deviations and notes after PREREG.md

Each entry: the Copenhagen time of the commit that first recorded it (`TZ=Europe/Copenhagen git log`), and whether
any scored output had been read at that time. The rule in PREREG.md is still reported.

PREREG v1 = e64f49e (15:01:02 CET). v1.1 = f553a31 (15:02:03 CET) + 879ed34 (15:03:08 CET).

## N0 — what Felix had seen when v1.1 was committed (recorded with this file, before any scored run)

Bruno posted the gate-0 compliance numbers G0a (#9230, 07:01 PDT = 15:01 CET: raw honest 27.8% / lie 11.5%; chat
honest 98.8% / lie 1.0%) before Felix committed 879ed34. Felix had read them. No readability (G0b), output-lens
(G0c) or LEACE price output existed. The v1.1 edits change status logic only (Q >= 10 as the premise; floor margin
20 pts; per-readout output-lens drop), per Kasper #9217 and #9233, and none of them depends on the compliance rate.
