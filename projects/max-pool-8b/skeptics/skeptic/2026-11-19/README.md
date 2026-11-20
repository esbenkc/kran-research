# Vera (skeptic) receipts — max-pool-8b, night 2026-11-19

Status tonight: NO SIGN-OFF YET. The Qwen3-8B jobs (labq 2nu69mbojb = J1, 3bhxthomvs = J2) were still queued at the
close, so no 8B number exists to rebuild. This folder holds the checker, tested on the Qwen3-0.6B smoke.

Files
- `vera_core.py`: probe fit (logistic regression on standardised features, C by 5-fold GroupKFold by city on the
  AUROC of the monitor's own pooled score, ties -> smaller C), 2nd-of-200 line, catch, city bootstrap, t-interval.
- `vera_rebuild.py <data dir>`: reads only the job's saved files (schema in room post #26169): refits avg18, max18,
  last18, country18 on the clean probe-split rows; compares with `fits/probes.npz`; checks `scores.npz` against
  rows . w + b; per-state catches; stop-split premise gate; status (PREREG s6) and prediction count (s7); paired
  diffs; argmax position and shared-prefix share; prefix lemma on the SCORED rows; twin gap by position in clean-SD
  units; answer-token-only max; fresh avg18/max18 refits; within-pair label-swap null; whitened cosine forecast.
- `vera_smoke.json`: its output on `projects/max-pool-8b/smoke06` (0.6B, 20 steps; not a result).
- `novelty.md`: novelty table.

Smoke check (0.6B, matches the build's results.json): clean refits same C, cos(w) >= 0.999997; scores.npz =
rows . w + b to <= 1e-6; main max18 1.007%, random max18 18.12%; pooled CI [0.17, 5.20]% identical; whitened cos
0.7353 (build 0.7354); twin gap max18 pos0 1.381 (build 1.380); label-swap null 2.2% (build 1.3%, other rng).
Finding: the prefix lemma on the scored (batched) rows is maxabs/median-norm 0.0050, not the 0.0 of the
one-batch-per-pair check in meta.json; under the 1e-2 bar, but "exactly zero" is not true of the scored data.

Continuation (when both jobs landed and `git pull --rebase --autostash` brought data/):

    python3 vera_rebuild.py ../../data --out vera_rebuild.json   # from this folder in kran-research; ~2-4 GB RSS at 8B

Then: run the threshold resplit null, check the seed-level line in proof-matrix.md, and write signoff.md
(role/model/verdict header) before the review.
