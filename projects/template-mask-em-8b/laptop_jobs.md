# Laptop-queue jobs (Esben's laptop, Apple M4 Pro 48 GB, MLX, Qwen/Qwen3-8B bf16)

Status as of 2026-11-21 14:45Z (`~/bin/labq status <id>`). Seed-2 cells of proof-matrix.md point here until their
answers and judge calls exist.

| job | room | cmd | outputs | status |
|---|---|---|---|---|
| 3zfskhbdya | theory-1118-template-mask-em-8b | run.sh p1 | out_p1/ | trained seed 0, passed mask checks; stopped by the memory guard in the first answer batch (DEVIATIONS C3) |
| fkyekw80z4 | theory-1118-template-mask-em-8b | run.sh p2 | out_p2/ | trained seed 1, passed mask checks; same stop |
| 57w02sk625 | theory-1119-template-mask-em | run_cont.sh ja | out_p1/, out_p2/ | done, exit 0, 1.71 h, peak 20.7 GiB; result commit 2a6558331 (base + seeds 0, 1, arms A-E) |
| dz6v4hu8p2 | theory-1119-template-mask-em | run_cont.sh jb | out_p3/ | failed before any work: git clone timed out (github.com:443, 396 s) |
| zh3iom3fww | theory-1121-template-mask-em | run_cont.sh jb | out_p3/ | cancelled 2026-11-21 before start; replaced by the split below (DEVIATIONS E1) |
| wzykhwatpb | theory-1121-template-mask-em | run_cont.sh jb1 | out_p3/ | queued (est 1.8 h, 28 GB): seed 2 train, mask check, arms A, B, E |
| 8h586zm3go | theory-1121-template-mask-em | run_cont.sh jb2 (TM_SRC_REF=swarm/theory-1121-template-mask-em) | out_p3/ | queued (est 0.5 h, 28 GB): seed 2 arms C, D from J-b1's adapter |

At the time of this file the worker esben-laptop was last seen 2026-11-21T02:21Z (asleep), with about 8.3 h of other
rooms' jobs ahead of J-b1.
