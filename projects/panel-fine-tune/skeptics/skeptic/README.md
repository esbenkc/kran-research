# Vera (skeptic) receipts — panel-fine-tune

Own code only (build's scripts not read). Run from this folder with the 1021 venv clone (`venv1021/bin/python`).
`vera_core.py` sets PROJ = this project and OLD = the theory-1021 full-stack-evader project (layer-18 attack acts, split, panel.npz).
Raw per-token activations (`data/acts/<state>/tok.npy`) are local only (sha256 in data/acts_sha256.txt).

- `vera_baseline.py` -> vera_baseline.json: max-panel on the 1021 states (clean, layer-18 attack, megaprobe attack, lambda 0).
- `vera_tokids.py` -> token_ids.json.
- `vera_check.py <tag> <states>` -> vera_check_<tag>.json: per-probe and max-panel catch, CIs, line nulls, lemma check, KL, 298 pairs.
  Tags: s012 (status arm, step 300), s012_150 (step 150), lam0, L18test (1021 layer-18 attack), L18lam0.
- `vera_anymember.py` -> exploratory any-member rule and re-standardised max (E1).
- `vera_refit.py` -> vera_refit_s01.json, vera_refit_s2.json: own refit on attacked activations + shuffled-label null.
- `vera_fwd.py` -> vera_fwd_panel__lam10000__seed2.json: own CPU forward of the seed-2 adapter.
- `vera_lemma.py` -> vera_lemma.json: readout-28 rescale lemma through the real unembedding; line-raise construction.
- `derivation.md`, `signoff.md`, `novelty.md`.
