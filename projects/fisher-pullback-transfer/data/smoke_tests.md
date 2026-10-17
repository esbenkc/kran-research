# Code smoke tests run before the pilot (not results; kept for the receipts)

1. 2026-10-17 ~15:06 CEST. Pilot probe `yahoo:Entertainment & Music`, lambda_KL 100, seed 0, 20 steps,
   PREREG loss (batch d'_w)^2 with the BATCH pooled sd. Target test set: w-score mean pos/neg
   3.36/0.19 -> 10.35/5.78, sd pos/neg 2.08/1.83 -> 12.00/9.05, gap 3.17 -> 4.57 (delta_w = +1.39),
   AUROC 0.874 -> 0.608, retain KL (held-out) 0.051 nats/token. 25 s, max RSS 630 MB.
   Consequence: DEVIATIONS.md 2026-10-17 15:13 (loss denominator fixed to the clean sd).
2. 2026-10-17 ~15:18 CEST. Target `dbpedia:NaturalPlace`, lambda_KL 100, seed 0, 10 steps, fixed-sd loss,
   only to test the new save code. Seen: 3 secondary probes 0.993 -> 0.635, 0.978 -> 0.626, 0.998 -> 0.499.
   No primary numbers were read. Output deleted, not used anywhere. 22 s, max RSS 745 MB.
3. 2026-10-17 ~15:55 CEST. Plumbing test of analysis.py on the UNUSED fixed-sd run (dbpedia:NaturalPlace seed 0,
   data/unused_fixed_sd/), copied to /tmp only. Seen (one target, unused loss, n=49 pairs): primary R^2 plain 0.07,
   whitened -5.9, Fisher -0.27; measured-drop SD 0.002, 0 pairs with |drop| > 0.02. Not used anywhere.
