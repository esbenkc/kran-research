# Vera (skeptic) receipts, cue-size-law-8b-rep

Written from PREREG.md and the raw run JSONs only. No code of the build was imported or reused.

| file | what it does |
|---|---|
| vera_check.py | rebuilds dense T from raw curves (gap = px_off - px_on, D = gap - gap(0), first eval with D >= 0.10 twice in a row), per-seed OLS slopes on ln share, seed-level t-interval, pooled Tobit, C1/C2 checks, S8 vs the parent 0.6B arm, S4 halves, size-permutation null, leave-one-size-out, prompt-level bootstrap (2,000 x 64 dev prompts, numpy seed 0) |
| vera_gates.py | G1 and REP-1 G2(b) from data/gates.json per-prompt numbers + the saved NDIF file (ties from its topv) |
| vera_g2a.py | G2(a) per token, per row and whole prompt, from the mini cache file and the saved NDIF block-34 input; cache array sha256 |
| vera_boot_diag.py | per-run distribution of the bootstrap T (why the prompt-level upper bound equals the point) |
| vera_band_sympy.py / .txt | slope band <-> per-halving factor arithmetic |
| vera_power0.py, vera_power5.py | pre-data power of the n=3 t-interval (Phase 0) |
| check_*.json | checker outputs (parent test, partial seeds, final) |
| novelty.md | novelty and definition check |
| signoff.md | verdict |

Run: `uv run --no-project --with numpy --with scipy python vera_check.py <project>/data/runs --prefix m8_ --seeds 5,6,7 --c1-seeds 5,6 --ref06 <parent runs_06b>`
(C1 s6 ran on the laptop as l8_c1_s6; the final check reads both prefixes, see signoff.md.)
