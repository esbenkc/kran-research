# cue-size-law-8b-rep

Replication of `cue-size-law-8b` (room theory-1120-cue-size-law-8b-rep). Frozen rule: `PREREG.md` (pushed
2026-11-20T14:01:06Z, header by `oracle.py prereg`). Changes after the push: `DEVIATIONS.md`.

## Where every cell ran

| cells | machine | job | output |
|---|---|---|---|
| cache, gates G1/G2, alphas, S9 scans | Mac mini | M1 (`laptop_job.sh` VARIANT=e7b_mini) | `data/gates.json`, `data/scan*.json`, `data/mini/M1/` |
| main arm seeds 5-7 x sizes k0-k5 (18 runs), C2 seed 5, C1 seed 5 | Mac mini | M1 | `data/runs/m8_*.json` |
| C1 seed 6 (first copy) + S9 seeds 5-7 | Esben's laptop | labq 5f3rwdo6um: runs finished (C1 s6 DONE 10:05:25 PST), job FAILED 10:27 PST on a heartbeat gap before it committed | not in git |
| recovery of 5f3rwdo6um's finished JSONs (copy only) | Esben's laptop | labq 9y8gn3qvoc (`recover_job.sh`), waits for the laptop | `data/runs/l8_*.json`, `data/runs_explore/`, `data/laptop/` |
| C1 seed 6 (mini copy, primary for controls) | Mac mini | M2 (`swarm-.../build/launch_mini_c1s6.sh`), started 10:31 PST, auto-commits at run end | `data/runs/m8_c1_s6.json`, `data/mini/M2/` |

Cancelled before any output: labq 7h93jdydad (full 3-lane job; laptop asleep). Failed before any output: labq
9jt0yykgi6 (3 heartbeat gaps).

## Rebuild

    uv sync --frozen --no-install-project
    .venv/bin/python analyze_8b.py      # -> data/analysis.json (frozen status rule + secondary reads)
    .venv/bin/python make_matrix.py     # -> proof-matrix.md
