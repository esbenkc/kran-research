# Vera (skeptic) receipts — max-pool-8b, night 2026-11-20

Verdict: PASS on the answer REJECTED (No). See `signoff.md`.

Own code only (numpy, scipy, sklearn); no build code read or run. Rebuild from this folder in kran-research:

    python3 vera_rebuild.py ../../data --out vera_rebuild.json        # ~18 min, ~4.4 GB RSS: refits, per-state catches, picks, status
    python3 vera_status_frozen_probes.py ../../data                     # 2 s: frozen-rule status with the build's frozen monitors
    python3 vera_nulls.py ../../data vera_nulls.json main__seed0__step150,main__seed1__step150,main__seed2__step150   # ~8 min
    python3 vera_whatif.py ../../data vera_whatif.json                  # pre-seed-2 bound (2 seeds + a hypothetical third)

Outputs: `vera_rebuild.json` (+ `.log`), `vera_status_frozen_probes.json`, `vera_nulls.json` (+ `.log`),
`vera_whatif.json`, `novelty.md`. Last night's (no-data) receipts: `2026-11-19/`.
