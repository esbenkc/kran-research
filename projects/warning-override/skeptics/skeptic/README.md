# Vera (skeptic) receipts — warning-override

Independent rebuild from the raw run logs (`../../raw/`), without build's scorer or analysis code.

    python3 vera_score.py ../../raw vera_counted_scores.jsonl          # own tags from PREREG s2
    python3 vera_check.py vera_counted_scores.jsonl ../../results/scores.jsonl > vera_check_counted.out
    python3 vera_robust.py vera_counted_scores.jsonl > vera_robust.out

Pre-data power sims: `vera_power0.py`, `vera_power0b.py` (phase 0, ceiling of the log-odds ratio), `vera_power1.py`
(edge choice), `vera_power2.py` (smoke rates). Verdict: `signoff.md`. Novelty: `novelty.md`. Matrix rows:
`proof-cells.md`. Python 3 + numpy only.
