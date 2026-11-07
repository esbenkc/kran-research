# Vera (skeptic) receipts — scoped-recording

Independent rebuild from the raw action logs (`../../raw/<model>/<arm>/<seed>.jsonl`) and PREREG v1.1 only.
Scripts were run from `swarm-defense-1107-scoped-recording/skeptic/` with the project at `../../kran-research/projects/scoped-recording`;
from here, pass `../../raw` (vera_check.py takes the raw dir as its argument) or edit the `R`/`P` constants.

- `signoff.md` — verdict table (status REJECTED reproduced), findings.
- `novelty.md` — prior-work check.
- `vera_check.py` -> `vera_check.out`, `vera_check.json` — own tagger, C1/C2, bootstrap, status, LOAO, shuffle null, K2, C3, P1-P4, D.
- `vera_extra.py` -> even/odd seeds, M1-passer subset, per-run M1 conditioning.
- `vera_manip.py` -> M1/M2 own parse; `vera_prompts.py` -> system prompts vs PREREG s1; `vera_null_tie.py` -> C2 null tie bug.
- `vera_raw_contrasts.out` — raw flag/key/any contrasts.
- `vera_power*.py/.out` — pre-data power of the status rule at the parent bases (n = 23-40).
- `vera_delivery.*` — word-count probe of replies; uninformative (the task itself says 'log level'), kept for the record.
