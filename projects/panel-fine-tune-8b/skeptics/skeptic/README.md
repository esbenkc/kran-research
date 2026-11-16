# Vera (skeptic) — receipts, panel-fine-tune-8b

Status 2026-11-16 06:05 CET: NO SIGN-OFF YET. The 8B jobs (labq lv07ii3whv, om3ghzbu47) are queued; no 8B data exists.
`signoff.md` is written only after the 8B raw data is rebuilt.

Own code (not build's scripts), written from PREREG.md s3 and D1.2:
- `vera_check.py`: max-panel catch at the state's own line (2nd-highest of 200 split-refit true), seed t-interval,
  city-cluster bootstrap (seed 1022), clean-line row. Validated on the 1.5B parent raw
  (swarm/theory-1022-panel-max-attack data/runs/panel__lam10000__seed*.npz): per seed 0.0218/0.0621/0.0285, pooled
  0.0375, city CI [0.0257, 0.0509], clean-line 0.980/0.953/0.961 = parent results.json exactly
  (`parent_rebuild.json`). Clean-line false alarms on eval true: 79.0/59.7/61.2% (mean 66.7%).
- `vera_rescale.py`: recalibration fix (1022 Rule B). Parent: 0.631/0.693/0.597, mean 0.640 (`parent_rescale.txt`).
- `vera_smoke_acts.py`: from saved activations, probe scores and panelmax equal build's score files (diff 0.0) and the
  last readout is model.norm(previous readout) (Qwen3-0.6B smoke, median rel err 0.0025).
- `novelty.md`: prior work, abs pages opened.

To run on 8B: point vera_check.py at the 8B score files (format differs: data/scores/<state>.npz has tok<R> raw
scores per state; adapt load()) and data/fits/zmoments.json; null tests to add then: line-resample threshold null,
and the per-seed spread vs the moving control.
