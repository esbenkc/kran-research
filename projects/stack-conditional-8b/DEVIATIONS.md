# DEVIATIONS — stack-conditional-8b

Every change to PREREG.md after its pushed prereg commit, and every departure of the run from it, is one dated line
here, with what had been seen when it was made.

## Before the freeze (no data of this run exists)

- P0. This PREREG.md supersedes PREREG-8B v2.1 of branch `scale/stack-conditional-8b` (commit 81aaef0f7, pushed
  2026-11-13). That design never trained: its only 8B data are untrained caches, gates and two 20-update smokes.
  Changes against it: the goal's thresholds (dGAP 10 points in 2 of 3 seeds, controls under 3 points in both
  always-penalised and both shuffled seeds; v2.1 had 0.05 and 0.02); the seed-level interval moves from the status to
  the report; the "recorder only" label; MLX dropped; emulated eval-only and last; the user-reward arm stays dropped;
  dev checks every 8 updates to 2,048, then every 64; the onset clause of section 4 is new.

## After the freeze
