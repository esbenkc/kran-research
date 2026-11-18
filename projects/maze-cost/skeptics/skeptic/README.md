# Vera (skeptic) receipts — maze-cost

Own code, written without reading the build's analysis code. Run from this folder:

- `python3 vera_check.py ../../data/<agent> --only maze|deadend [--prices in,out]` — rebuilds entry, depth, leaves, spend,
  the frozen pilot rule (h, d*, P, KM O, R, window), the constancy LR test, leave-rate bounds and the dollar ratio from
  `raw/<arm>/*.json` (steps[].meta tokens + actions) and `access/<arm>/*.jsonl` (server log). Outputs: `out_*.json`.
- `python3 vera_nulls.py ../../data/<agent> ...` — paired sign-flip null (maze vs dead end), price swap, cache trap,
  stuck check, harness agreement.
- `python3 vera_gate.py ../../data/<agent>` — gate count from transcripts.
- Pre-data power: `vera_law_power.py`, `vera_window.py <K>`, `vera_mean.py <K> <n>`, `vera_lr.py`; `vera_synth.py` makes
  synthetic raw data to unit-test the checker.
- `handcode/` — blind 'noticed the endless maze' coding (packet, key, two coders, result.md).
- `signoff.md` (verdict), `novelty.md` (prior work).
