# Vera (skeptic) — independent rebuild

Own code only (numpy/scipy/sklearn); nothing imported from the build scripts. Run from the work dir that holds `kran-research/`:

    uv run --with numpy --with scipy python kran-research/projects/rethreshold-probe-evasion/skeptics/skeptic/vera_rebuild.py kran-research/projects/rethreshold-probe-evasion/data/runs --draws 2000
    uv run --with numpy --with scipy python .../vera_theory_test.py kran-research/projects/rethreshold-probe-evasion/data/runs
    uv run --with numpy --with scipy --with scikit-learn python .../vera_theory_check.py
    uv run --with numpy --with scikit-learn python .../vera_gate.py
    uv run --with numpy --with scikit-learn python .../vera_lam0_probe.py   # needs the local, gitignored data/runs/*__acts.npz

- `signoff.md` — verdict, numbers, nulls, conditions on the post.
- `derivation.md` — my own derivation of the binormal TPR, the re-fit share and Felix's E6 limits.
- `novelty.md` — sources checked and what is and is not new.
- `vera_rebuild_full.json` — every per-run and pooled number (data commit faa54c5).
