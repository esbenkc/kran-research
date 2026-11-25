# DEVIATIONS — llc-bias-law

Each entry: date-time (PST), who, what changed against PREREG.md, why, and how much it can move the result.

## Pre-freeze pilots (before the PREREG push; disclosed, not deviations)
- 2026-11-25 05:58-06:04, Kasper: rebuild of the parent's `data/gate/grid.json` (scout number X fails at matched X).
- 2026-11-25 05:59-06:02, Vera: `skeptic/vera_isoX.py`, `vera_isoG.py`, `vera_isoGnb.py`, `vera_gibbs.py`,
  `vera_split.py` on the training shapes (32,32,16) r1, r8 only.
- 2026-11-25 06:00-06:04, Bruno: `scratch/sgld_batch.py` (timing, Gibbs), `scratch/mix.py` (T pilot) on the training
  shapes only.
- 2026-11-25 06:03, Vera: `skeptic/vera_power_v2.py` -> `vera_power_v2_1500_2.json`: exact Gibbs on all drafted
  held-out shapes (now the SEEN shapes of PREREG 2(5)) on an earlier draft grid. Those shapes carry no status weight;
  the status rests on five fresh BLIND shapes that nobody ran before predictions.json is pushed.
- Choices made after Vera's 06:03 pilot, each with its reason (all before the PREREG push): status moved to five
  fresh BLIND shapes (Tove #30704, Kasper #30703); nbeta 10 out (training shape (32,32,16,8) misses +9..+17 points
  there, Vera #30700); psi(q) form isotonic -> 1/(1+(a/q)^b) (Felix #30698); secondary law L_z added and labelled
  "hypothesis formed after the pilot" (z from Kasper's pre-pilot Gaussian limit #30691); q < 0.1 secondary (Tove
  #30692, posted before the pilot).
