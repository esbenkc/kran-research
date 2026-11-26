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

- D1 (2026-11-26 02:30 CET, Iris; seen: no 8B data, job mdnk6nplc4 still queued). PREREG s2 says the emulated eval
  runs "after data/status.json is written". The job (run_job.sh @24b13e27a) does not write status.json; analyze.py
  (0826a711b) writes it after the job from the job's outputs. The job runs eval_emu only after every whole-graph and
  tail-level status input exists (chain.log line "STATUS INPUTS READY"), so the emulated eval still cannot delay or
  change the status inputs. No rule, threshold or metric changes.
- D1 resolved (2026-11-26 02:35 CET, Iris; seen: no 8B data). Bruno cancelled mdnk6nplc4 while queued (it never
  started) and resubmitted as mx2tsl4i17 (commit caf2a8265): the job now runs analyze.py to data/status.json BEFORE
  eval_emu, as PREREG s2 says, and g1b.py writes data/g1b.json (PREREG s5 G1b). The job now matches PREREG s2 and s5.
- D2 (2026-11-26 02:40 CET, Iris; seen: no 8B data). Job ids: mdnk6nplc4 -> mx2tsl4i17 -> cc1jn7qx0o -> eukptwkl12
  (commit 91a45216c). Each was cancelled while queued (none started: worker last seen 2026-11-25T21:49Z). The later
  ones only add outputs for rebuilds (data/cache_meta/, data/rows02_dev.npz); no rule, metric or step order changed.
