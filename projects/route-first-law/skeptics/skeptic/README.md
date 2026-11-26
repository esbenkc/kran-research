# Skeptic receipts (Vera)

Own code; nothing imported from the build. Run from this folder with the project dir as argument.

- `vera_rebuild.py <project>/data/main` -> `out_rebuild.txt`: frozen-rule status from raw logs (REJECTED).
- `vera_null.py` (reads `rebuild_rows.json`) -> `out_null.txt`: paired McNemar, label-swap permutation, block and model splits.
- `out_tail.txt`: binomial tails for the AFTER cells under a true 75%.
- `vera_t3_web.py`, `vera_t3_shell.py <project>` -> `out_t3_*.txt`: pre-data arm checks (bytes, order, no-trap reachability).
- `vera_fixture.py <project> <tmp>` -> `out_fixture.txt`: 11 planted-edge fixtures, my checker vs analysis.py.
- `vera_power_*.py` -> `out_power_*.txt`, `out_guard_equiv.txt`: rule power before the freeze.
- `vera_parent_order.py`, `out_wilson_parent.txt`: phase-0 recount of the parent nofollow-canary data.
- `signoff.md`, `novelty.md`.
