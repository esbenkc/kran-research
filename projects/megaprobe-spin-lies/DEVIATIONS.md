# DEVIATIONS — megaprobe-spin-lies

Changes after the PREREG.md freeze (commit 09ffe13, 02:01:56 CEST). Each entry's time is the git commit time of the
commit that adds it (`git log --format=%cd -- DEVIATIONS.md`), never a typed guess.

## D1 — arm (c2): FIBS axial probe (added before any probe fit)

Why: after the freeze, Iris found the newer public Cundy/Gleave multi-layer recipe: Hollinsworth, Spies, Diriba,
Gleave, Cundy, "Telling FIBS" (arXiv 2610.12445), code github.com/AlignmentResearch/fibs-probes @6cae61d
(src/probe_inference/archs/axial.py, App. A.3). Our arm (c) is close to it but is our own build. A NO on (c) alone is
open to "the megaprobe was weak". No probe had been fit on this run's activations when this entry was committed
(J2 had not started; room #3419, #3424).

What:
- New arm (c2): FIBS AxialProbe, code defaults (d_proj 128, 4 heads, 1 block), layers 7, 10, 13, 16, 19, 22 (block
  outputs), score at the last read token capped 10*tanh(x/10). Read window = the answer tokens plus the cached final
  token after the follow-up turn (the follow-up turn's other tokens are not in the J1 cache; Bruno states the exact
  window in the code docstring). Same folds, same draws, same label sets as (c). Hyperparameters (learning rate,
  weight decay, epochs) chosen only by inner CV on the base model's training folds, frozen before the first trained-
  model fit, like (c).
- Status rule (PREREG s6) changes only on the NO side:
  - YES (CONFIRMED): unchanged, on (c) alone (m >= 0.80 and the 95% lower bound of g > 0). (c2) gives no second
    chance at a YES.
  - NO (REJECTED): the 95% upper bound of m < 0.80 for BOTH (c) and (c2), where m for (c2) is the same mean over the
    4 trained models of its spin-vs-fact-stated-truthful AUROC.
  - UNCONFIRMED otherwise. Guards (all-truthful negatives, positive control) apply to (c2) as to (c).
- Drop rule: if (c2) is not scored on all 5 models by 04:00 CEST, (c2) is dropped, the NO rule reverts to PREREG s6
  ((c) alone), and the post states the caveat once: FIBS trained its probe on ~153k rows, we fit on 400.
- The prediction outcome (PREREG s7) is scored on (c) only, as written; (c2) values are printed next to it.

## D2 — arm (b) input scaling follows the public code (added before any probe fit)

Why: PREREG s5(b) says "unit-normalise each final-token vector". The public code does something else: its
'unit_norm' mode divides every vector by one per-layer scalar, sqrt(mean ||x||^2) over the training rows
(obfuscation-atlas probe_archs.py _maybe_normalize, train.py:178 compute_input_scales; FIBS 2610.12445 App. A.1 uses
the same per-layer RMS scale). Per-vector normalisation is the code's separate 'l2' mode (Iris, room #3425).
What: arm (b) uses the per-layer RMS scale, fit on each training draw only. The per-vector L2 version, as written in
PREREG, is reported next to it as the sensitivity line "b_l2". Nothing else in (b) changes. No probe had been fit when
this entry was committed.

## D3 — the (c2) drop line moves from 04:00 to 04:30 CEST (added before any (c2) fit on a trained model)

Why: (c2) needs 3.2 GB on the GPU or 1.96 GB on CPU, and the (c) fits need 3.35 GB, so the two jobs cannot run at
once under the 5 GB rule (room #3443, #3446). Run serially, (c2) ends about 04:25.
What: the drop line in D1 moves to 04:30 CEST. If (c2) is not scored on all 5 models by then, D1's drop rule applies
unchanged. The stop rule (PREREG s9, 05:30) does not change. No (c2) score on any model existed when this entry was
committed.
