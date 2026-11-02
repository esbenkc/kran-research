# Vera (skeptic) receipts — open-route

Own rebuild from `../../raw/` only (build's scorer.py / analysis.py not read). Run from this folder:

    python3 vera_score.py ../../raw vera_counted_scores.jsonl      # per-run tags (PREREG s2 + D2)
    python3 vera_check.py vera_counted_scores.jsonl 10000 > vera_check_counted.out   # s5 + D5 status, s6 law, s7 prediction
    python3 vera_null.py vera_counted_scores.jsonl > vera_null_counted.out           # permutation null, LOMO, seed halves, cluster CI
    python3 vera_split.py ../../raw > vera_split.out                                 # key use by first storage reply (floor contamination)

Pre-data: vera_power0.py, vera_power_status.py, vera_power_frozen.py, vera_power_d3.py (+ .out). Smoke: vera_smoke*_scores.jsonl.
Verdict: signoff.md. Novelty: novelty.md.
