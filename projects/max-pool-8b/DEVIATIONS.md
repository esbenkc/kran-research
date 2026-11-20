# Deviations — max-pool-8b

Every change after the PREREG freeze (commit 335ccd976, pushed 2026-11-19T00:58:16Z). Times from `git log`.

## D1. Label fix in PREREG s9 item 3 reference numbers (no rule change; before any 8B data)

PREREG.md s9 item 3 quotes the 1.5B twin gaps as "`frozen` 4.56 -> 0.92, `max14` 5.13 -> 0.42". Both pairs belong to
`max14`: position 0 (the country) 4.557 -> 0.918, position 1 (the full stop) 5.133 -> 0.422. The `frozen` (averaging
probe) rows are 4.040 -> 0.720 (position 0) and 4.677 -> 0.265 (position 1). Source: max-pool-survives
`data/results/results.json` `diagnostics.separation_by_rel_pos` (seed mean = `attacked_mean`). Iris printed
`max14['0']` and `max14['1']` and labelled the first as `frozen`; Selma caught it (#26145), Kasper confirmed (#26147).
The definition in s9.3 is unchanged; s9.3 is a diagnostic outside the FROZEN RULE block. The reference values are:

| 1.5B probe | position 0 (country) | position 1 (full stop) |
|---|---|---|
| `frozen` (averaging) | 4.040 -> 0.720 | 4.677 -> 0.265 |
| `max14` | 4.557 -> 0.918 | 5.133 -> 0.422 |

## D2. Seed-1 gate files written on the mini; seed 1's step-300 extract skipped (no rule change; 2026-11-20 room)

Laptop job J1 `btb9dgh9bw` stopped on the laptop's disk guard (0.2 GB free disk) during main seed 1's step-300
scores-only extract, after seed 1's step-150 extract and `premise gate step 150 main__seed1: pass` (data/logs/J1/chain.log).
The frozen gate scores step 150 when its stop catch is at most 10%, so step 300 is never scored for seed 1. In room
oracle-1120-max-pool-8b-cont Bruno ran `gate.py final main 1` (stop catch 0.01 at step 150, scored_step 150,
checkpoints["300"] = null) and then `gate.py final random 1` (follows the partner: step 150; prunes random seed 1's
step-300 tok shards) on the Mac mini. gate.py is numpy-only (no model), so the files equal what the laptop would write.
Main seed 2 runs as laptop job J3 (`./run_all.sh clean 'main 2'`, same code, same fits).
