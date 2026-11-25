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

## After the PREREG push (2026-11-25 14:07:04Z)
- 06:2x PST, Bruno (#30730, disclosed): a harness smoke test ran ONE off-grid setting on the blind shape (24,24,12,3):
  arm S, nbeta 100, q 0.5 (not a grid value), c 0.1, seeds 0-1, T 1e4: SGLD e about -15%, Gibbs -16.8/-17.1%. Files
  moved to the build scratch folder `scratch/peek-disclosed/` (not data). The frozen blind list is unchanged (the
  pushed file is the record; Felix #30737). Remedy: score.py reports the status on the frozen 180 cells (the status)
  and, as a sensitivity row, without (24,24,12,3); if the two disagree, the post says so.
- Iris, fact correction (Kasper #30731): PREREG 2(1) says the sample covariance of arm S moves q, z and h_i "by about
  1/sqrt(n) relative". That is wrong for the extreme Hessian eigenvalues: at n = 1024 the data-seed Hessian of K_n at
  w* has regular eigenvalues 0.67-1.93 (r1) and 0.71-2.37 (r8) against the population values {1, 2}
  (Marchenko-Pastur spread, sqrt(M/n) ~ 0.18). The rule itself is unchanged (the law and eps use the population
  values, as frozen). Measured impact on the law's regular term, training shapes, all 24 primary settings each
  (`research/hess_sample_cov.py`, `hess_sample_cov.json`): -0.30 to -0.002 points of lambda, median -0.02. The step
  size uses the population h_max = 2, so the realised c is up to 18% above nominal (0.1 -> 0.118); that moves the step
  factor 1/(1 - c/4) by about 0.5% of the regular part. Both are far below the 5-point bar.
- 2026-11-25 08:17-08:20 PST, Selma (per Tove #30977): critic2 (openai/gpt-5.5) gave its PASS at 16:17:26Z (room #30969), then hit its
  $2.44 spend cap ($2.96 used) and swarm restarted the role on Claude at 16:18:39Z (#30972), which rewrote the swarm record
  critic2.family. The Claude fallback gave no verdict. The gpt-5.5 receipts moved to skeptics/critic2-gpt55/ (role critic2-gpt55)
  so the sign-off keeps the model that wrote it; the swarm record was not edited. No rule, threshold or metric changed.
