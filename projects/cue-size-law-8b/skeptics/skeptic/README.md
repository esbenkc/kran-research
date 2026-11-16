# Vera (skeptic) receipts

Sign-off: signoff.md (PASS on the numbers; status UNCONFIRMED, EXPLORATORY). All scripts are my own and read only raw run JSONs.

- vera_check.py: pre-data checker, validated on the parent (dense slope -0.90265).
- vera_power.py/.out: power of the frozen status rule (pre-data).
- vera_t3.py: port regression vs parent runs (bit-equal curves).
- vera_scan.py/.json: cue size x and share rebuilt from NDIF rows + safetensors norm weight.
- vera_g2_floor.py/.json: NDIF determinism at a fixed batch size.
- vera_final.py/.json: T, per-seed slopes, seed-level CI, S1, S2, permutation null, controls (usage: python3 vera_final.py <project dir> <parent runs dir>; parent runs = git show origin/swarm/theory-1110-cue-size-learning:projects/cue-size-law/data/runs/).
- vera_robust.py/.out/.json: drop-one, halves, start-up offset, quadratic bend.
- novelty.md: prior work check.
