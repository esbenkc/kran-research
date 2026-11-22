# Vera (skeptic) receipts, cue-size-law-8b-rep

Written from PREREG.md and the raw run JSONs only. No code of the build was imported or reused.
Nov 20 sign-off and its outputs: ../2026-11-20/skeptic/. Tonight (Nov 22, continuation): signoff.md.

| file | what it does |
|---|---|
| vera_check.py | rebuilds dense T from raw curves, per-seed OLS slopes, seed-level t-interval, Tobit, C1/C2, nulls, prompt bootstrap |
| check_final.json | its output on the final branch (prefixes m8_, l8_) |
| vera_diff.py / .txt | field-by-field diff of data/analysis.json status_8b against check_final.json (70 fields) |
| vera_c1s6_provenance.py / .txt | C1 s6 test gap vs the pre-failure log line; killed mini rerun M2 vs laptop copy (47/47) |
| vera_seed_null.py / .txt | size-label shuffle null of the seed-level interval; width with the Nov 16 and 0.6B seed sd |
| vera_s9.py / .txt, check_s9.json | S9 fresh-direction slopes with the frozen gap(0) exclusion |
| vera_matched.py / .txt | main seeds refit on each S9 seed's sizes |
| vera_m3.py / .txt | M3 checks: C2 s6, S9 k5 s7 args, mini vs partial laptop log (15/15) |
| vera_gates.py, vera_gates_1122.txt | G1 and G2 from data/gates.json + the saved NDIF file |
| vera_g2a.py, vera_boot_diag.py, vera_band_sympy.py/.txt, vera_power*.py | Nov 20 checks (unchanged inputs) |
| novelty.md | novelty and definition check (Nov 20 + Nov 22 refresh) |
| signoff.md | verdict |

Run (from the project dir): `uv run --no-project --with numpy --with scipy python skeptics/skeptic/vera_check.py data/runs --prefix m8_,l8_ --seeds 5,6,7 --c1-seeds 5,6 --gates data/gates.json --out check_final.json`,
then from skeptics/skeptic: `python vera_diff.py ../../data/analysis.json check_final.json`, `python vera_seed_null.py`,
`python vera_s9.py ../..`, `python vera_matched.py`, `python vera_m3.py ../..`.
