# Vera (skeptic) receipts — decoy-usefulness

Own code, written from PREREG.md / DEVIATIONS.md and the raw JSONL only (build's analysis.py is not imported).
numpy + scipy, < 200 MB, seconds. Run from this folder (projects/decoy-usefulness/skeptics/skeptic/):

    python3 vera_check.py ../../raw --seeds 0-19 --out vera_counted.json   # rebuild: cells, U, R, D, CI, status, nulls
    python3 vera_exact.py      # exact run-level bootstrap of D (convolution), RNG-free status check
    python3 vera_sens.py       # status under each pre-data choice, leave-one-out, label-flip nulls, variants
    python3 vera_harness_bb.py # black-box harness check (T2): cells differ only in the record slots
    python3 vera_power.py; python3 vera_power2.py; python3 vera_power3.py   # pre-data power sims (phase 0)

smoke_check.json = the same checker on raw_smoke (not counted). Verdict: signoff.md. Novelty: novelty.md.
