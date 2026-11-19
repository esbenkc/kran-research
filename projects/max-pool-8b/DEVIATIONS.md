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
