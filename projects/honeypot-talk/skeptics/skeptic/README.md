# Vera (skeptic) receipts — honeypot-talk

Own code, written from PREREG.md (v1.2/v1.3) only; no code from `scorer.py` or `analysis.py` was read or imported.
Run from this folder with the project's raw data at `../../raw` (paths in the scripts point to the room work dir;
replace `../../kran-research/projects/honeypot-talk/raw` with `../../raw`). numpy + scipy, CPU, < 300 MB, seconds.

Pre-data (simulated data only):
- `vera_power.py`, `vera_power_n.py`, `vera_power_edge.py`, `vera_power_alt.py`, `vera_power_felix.py`,
  `vera_power_frozen.py` (+ `.out`): TOST/status power and size; showed the v1 two-level CI made Yes unreachable
  (P(Yes | G=0) 0.05), which led to D2 (within-stratum CI).
- `vera_power_frozen.py smoke_rates_v13_N50.json / _N70.json` -> `vera_power_smoke_N50.out`, `_N70.out`: the v1.3
  power table (N_on 50: Yes|0 0.13; 70: Yes|0 0.53, No|-0.25 0.95, edge errors 0.05).
- `vera_spearman_exact.py`: exact null of Spearman for n = 5 and 10.

Counted data (1486f18a):
- `vera_score.py RAW OUT.jsonl`: own scorer (exposure, window, trip, says variants incl. v1.3 primary `says_v13`).
- `vera_check.py vera_counted_scores.jsonl says_v13 vera_check_primary.json`: G, W/M/F2 CIs, s6 status,
  within-stratum shuffle null, placebo regexes, pooled row, s7, s8, Spearman -> `vera_check_primary.out`.
- `vera_robust.py`, `vera_robust2.py`: leave-one-model-out, all strata, variants, seed halves, placebos on the frozen
  strata, v1.2 vs v1.3 per stratum.
- `vera_s7.py`: prediction outcome with 90% CIs.
- `vera_keytarget.py`: own build of the named key_targeted row. `vera_keytarget_nosemicolon.out` = the script as
  committed (sentences split at . ! ?): G -0.279; `vera_keytarget.out` = the earlier run with ';' also a split: -0.363.
- `signoff.md` (verdict and numbers), `novelty.md` (prior work).
